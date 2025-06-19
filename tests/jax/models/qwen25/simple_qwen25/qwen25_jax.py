#!/usr/bin/env python3
"""
qwen25_jax.py – minimal single-device JAX inference for Qwen-2.5-7B

Usage:
python qwen25_jax.py \
       --model_path ../weights \
       --prompt "Explain the link between AI and consciousness." \
       --max_new_tokens 256 \
       --dtype bfloat16
"""
import argparse, json, time, gc, os, functools, numpy as np, jax, jax.numpy as jnp
from flax import linen as nn
from safetensors import safe_open
from transformers import AutoTokenizer
# --------------------------------------------------------------------------------------
# 0.  Tiny helpers
# --------------------------------------------------------------------------------------
def sample_top_p(logits, *, top_p=0.9, top_k=50, temperature=0.7, key):
    """Return one token id per batch element with nucleus + top-k filtering."""
    if temperature > 0:
        logits = logits / temperature
    # Top-k first (much cheaper than sort every time)
    if top_k and top_k < logits.shape[-1]:
        top_k_logits, top_k_idx = jax.lax.top_k(logits, top_k)
        mask = jnp.full_like(logits, -jnp.inf).at[top_k_idx].set(top_k_logits)
        logits = mask
    # Top-p
    probs   = jax.nn.softmax(logits, axis=-1)
    sort_p, sort_idx = jax.lax.sort_key_val(-probs, jnp.arange(probs.shape[-1]))
    sort_p = -sort_p
    cdf    = jnp.cumsum(sort_p, axis=-1)
    cut    = jnp.argmax(cdf > top_p, axis=-1)[..., None]
    mask   = jnp.broadcast_to(jnp.arange(probs.shape[-1]), sort_idx.shape) > cut
    probs  = probs.at[jnp.take_along_axis(sort_idx, mask, axis=-1)].set(0.0)
    probs  = probs / jnp.sum(probs, axis=-1, keepdims=True)
    return jax.random.categorical(key, jnp.log(probs), axis=-1)

def repeat_kv(x, repeat):
    """(b, s, kv_h, d) → (b, s, kv_h*repeat, d) without copy."""
    b, s, h, d = x.shape
    x = x[:, :, :, None, :].repeat(repeat, axis=3)
    return x.reshape(b, s, h * repeat, d)
# --------------------------------------------------------------------------------------
# 1.  Model – only the pieces required for forward / incremental decode
# --------------------------------------------------------------------------------------
class QwenAttention(nn.Module):
    config: dict; dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.h    = c["num_attention_heads"]
        self.kh   = c.get("num_key_value_heads", self.h)
        self.d    = c["hidden_size"] // self.h
        self.q = nn.Dense(c["hidden_size"],             dtype=self.dtype, name="q_proj")
        self.k = nn.Dense(self.kh*self.d,               dtype=self.dtype, name="k_proj")
        self.v = nn.Dense(self.kh*self.d,               dtype=self.dtype, name="v_proj")
        self.o = nn.Dense(c["hidden_size"], use_bias=False, dtype=self.dtype, name="o_proj")
        self.theta = c.get("rope_theta", 1e4)

    def __call__(self, x, *, pos_ids, mask, cache):
        b, s, _ = x.shape
        q = self.q(x).reshape(b, s, self.h,  self.d)
        k = self.k(x).reshape(b, s, self.kh, self.d)
        v = self.v(x).reshape(b, s, self.kh, self.d)
        cos, sin = rotary_embedding(self.d, pos_ids, self.theta)
        q, k = apply_rope(q, k, cos, sin)

        # past-kv: (b, past, kh, d)
        pk, pv = cache or (jnp.zeros((b,0,self.kh,self.d), dtype=q.dtype),
                           jnp.zeros((b,0,self.kh,self.d), dtype=q.dtype))
        k = jnp.concatenate([pk, k], axis=1)
        v = jnp.concatenate([pv, v], axis=1)
        cache = (k, v)

        if self.h != self.kh:   # GQA
            repeat = self.h // self.kh
            k_r, v_r = repeat_kv(k, repeat), repeat_kv(v, repeat)
        else:
            k_r, v_r = k, v

        q = jnp.transpose(q, (0,2,1,3))
        k_r = jnp.transpose(k_r, (0,2,1,3))
        v_r = jnp.transpose(v_r, (0,2,1,3))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_r) / jnp.sqrt(self.d)
        scores = scores + mask
        probs  = jax.nn.softmax(scores, axis=-1)
        out    = jnp.einsum("bhqk,bhkd->bhqd", probs, v_r)
        out    = jnp.transpose(out, (0,2,1,3)).reshape(b, s, -1)
        return self.o(out), cache

def rotary_embedding(dim, pos, theta):
    inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = pos[:, :, None] * inv            # (b, s, dim/2)
    emb   = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
    return emb, emb                # cos==sin wallet-splitting trick ;-)

def apply_rope(q, k, cos, sin):
    def rot(x):
        x1, x2 = jnp.split(x, 2, -1)
        return jnp.concatenate([-x2, x1], -1)
    q = q * cos[...,None,:] + rot(q) * sin[...,None,:]
    k = k * cos[...,None,:] + rot(k) * sin[...,None,:]
    return q, k

def causal_mask(q_len, k_len, dtype):
    i = jnp.arange(q_len)[:,None]
    j = jnp.arange(k_len)[None,:]
    mask = (i < j - (k_len - q_len))
    return mask.astype(dtype) * -1e9

class QwenMLP(nn.Module):
    config: dict; dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        h = c["hidden_size"]; inter = c.get("intermediate_size", 4*h)
        self.gate = nn.Dense(inter, dtype=self.dtype, name="gate_proj")
        self.up   = nn.Dense(inter, dtype=self.dtype, name="up_proj")
        self.down = nn.Dense(h,    dtype=self.dtype, name="down_proj")
    def __call__(self, x):
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))

class DecoderLayer(nn.Module):
    config: dict; dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.attn = QwenAttention(self.config, dtype=self.dtype)
        self.mlp  = QwenMLP(self.config, dtype=self.dtype)
        eps = self.config.get("rms_norm_eps", 1e-5)
        self.ln1 = nn.RMSNorm(eps, dtype=self.dtype)
        self.ln2 = nn.RMSNorm(eps, dtype=self.dtype)
    def __call__(self, x, *, pos_ids, mask, cache):
        h, cache = self.attn(self.ln1(x), pos_ids=pos_ids, mask=mask, cache=cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, cache

class QwenFlax(nn.Module):
    config: dict; dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.embed = nn.Embed(c["vocab_size"], c["hidden_size"], dtype=self.dtype, name="embed_tokens")
        self.layers = [DecoderLayer(c, dtype=self.dtype, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.ln_f  = nn.RMSNorm(c.get("rms_norm_eps",1e-5), dtype=self.dtype)
        self.head  = nn.Dense(c["vocab_size"], use_bias=False, dtype=self.dtype, name="lm_head")

    def __call__(self, input_ids, *, pos_ids, mask, caches):
        x = self.embed(input_ids)
        new_caches=[]
        for layer, cache in zip(self.layers, caches):
            x, cache = layer(x, pos_ids=pos_ids, mask=mask, cache=cache)
            new_caches.append(cache)
        x = self.ln_f(x)
        return self.head(x), new_caches
# --------------------------------------------------------------------------------------
# 2.  Weights loader  (unchanged mapping logic but stripped of logging & safety checks)
# --------------------------------------------------------------------------------------
def qwen_path(hf_name):
    if hf_name.startswith("model.layers."):
        # Parse layer weights: model.layers.{i}.{component}.{param}
        parts = hf_name.split(".")
        layer_idx = int(parts[2])
        component = parts[3]
        param = parts[4]
        
        if component == "self_attn":
            if param == "q_proj":
                return (f"layers_{layer_idx}", "attn", "q", "kernel")
            elif param == "k_proj":
                return (f"layers_{layer_idx}", "attn", "k", "kernel")
            elif param == "v_proj":
                return (f"layers_{layer_idx}", "attn", "v", "kernel")
            elif param == "o_proj":
                return (f"layers_{layer_idx}", "attn", "o", "kernel")
        elif component == "mlp":
            if param == "gate_proj":
                return (f"layers_{layer_idx}", "mlp", "gate", "kernel")
            elif param == "up_proj":
                return (f"layers_{layer_idx}", "mlp", "up", "kernel")
            elif param == "down_proj":
                return (f"layers_{layer_idx}", "mlp", "down", "kernel")
        elif component == "input_layernorm":
            if param == "weight":
                return (f"layers_{layer_idx}", "ln1", "scale")
        elif component == "post_attention_layernorm":
            if param == "weight":
                return (f"layers_{layer_idx}", "ln2", "scale")
        return None
        
    return {
        "model.embed_tokens.weight": ("embed_tokens","embedding"),
        "model.norm.weight"       : ("ln_f","scale"),
        "lm_head.weight"          : ("lm_head","kernel")
    }.get(hf_name)

def load_safetensors(folder, model, dtype):
    num_layers = model.config["num_hidden_layers"]
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1,1), jnp.int32),
                        pos_ids=jnp.ones((1,1), jnp.int32),
                        mask=jnp.zeros((1,1,1,1), dtype),
                        caches=[None]*num_layers)
    flat = {}
    for f in os.listdir(folder):
        if not f.endswith(".safetensors"): continue
        with safe_open(os.path.join(folder,f), framework="numpy") as st:
            for name in st.keys(): flat[name]=st.get_tensor(name)
    # map
    def insert(tree, path, val):
        for p in path[:-1]:
            tree = tree.setdefault(p, {})
        tree[path[-1]] = jnp.asarray(val, dtype=dtype)
    new={}
    for k,v in flat.items():
        path = qwen_path(k)
        if path: insert(new, ("params",)+path, v.T if v.ndim==2 and "proj" in k or "lm_head" in k else v)
    return jax.tree_util.tree_map(lambda p,n: n if n is not None else p, params, new)
# --------------------------------------------------------------------------------------
# 3.  Incremental text generation (single-batch)
# --------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("model","params","steps"))
def incremental_decode(model, params, input_ids, *, steps, temperature, top_p, top_k, rep_pen):
    b = input_ids.shape[0]
    caches = [None]*len(model.layers)
    seq_len = input_ids.shape[1]

    def body(state, t):
        ids, caches, pos = state
        # mask is always 1 (no padding) → causal_mask handles autogressivity
        mask = causal_mask(1, pos+1, jnp.float32)[None,None,:,:]
        logits, caches = model.apply(params, ids, pos_ids=jnp.array([[pos]]),
                                     mask=mask, caches=caches)
        logits = logits[:, -1, :]
        logits = logits / rep_pen  # repetition penalty (>1.0) favours new tokens
        key    = jax.random.fold_in(jax.random.PRNGKey(0), t)
        next_id = sample_top_p(logits, top_p=top_p, top_k=top_k,
                               temperature=temperature, key=key)
        return (jnp.array([[next_id]]), caches, pos+1), next_id
    (_, _, _), toks = jax.lax.scan(body, (input_ids, caches, seq_len-1), None, length=steps)
    return toks.T.squeeze(0)   # (steps,)
# --------------------------------------------------------------------------------------
# 4.  CLI
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--dtype", choices=["bfloat16","float32"], default="bfloat16")
    args = p.parse_args()

    dtype = jnp.bfloat16 if args.dtype=="bfloat16" else jnp.float32
    cfg   = json.load(open(os.path.join(args.model_path,"config.json")))
    model = QwenFlax(cfg, dtype=dtype)
    print("Loading safetensors …")
    params = load_safetensors(args.model_path, model, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # chat template (for *-Instruct* checkpoints)
    messages = [{"role":"user","content":args.prompt.strip()}]
    ids = tokenizer.apply_chat_template(messages,
                                        tokenize=True,
                                        add_generation_prompt=True,
                                        return_tensors="np").input_ids
    print(f"Prompt tokens: {ids.shape[1]}")
    t0 = time.time()
    new_ids = incremental_decode(model, params, ids,
                                 steps=args.max_new_tokens,
                                 temperature=args.temperature,
                                 top_p=args.top_p,
                                 top_k=args.top_k,
                                 rep_pen=args.repetition_penalty)
    out = tokenizer.decode(np.concatenate([ids[0], np.asarray(new_ids)]),
                           skip_special_tokens=True)
    print(f"\n--- generated in {time.time()-t0:.1f}s ---\n")
    print(out)

if __name__ == "__main__":
    main()
