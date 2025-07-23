# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
from .model import QwenModel, KVCache, Axis, Qwen25ForCausalLM
from .util import sample_top_p, timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="[INST]What is the name of the largest planet in our solar system?[/INST]")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # SHARDING_RULES = list(
    #     {
    #         Axis.EMBED: None,
    #         Axis.MLP: "x",
    #         Axis.HEAD: "x",
    #         Axis.QHEAD: None,
    #         Axis.KVHEAD: None,
    #         Axis.VOCAB: None,
    #     }.items()
    # )

    # mesh = jax.sharding.Mesh(jnp.array(jax.devices('cpu')).reshape(-1, 1), ("x",))

    model = Qwen25ForCausalLM.load(args.model_dir, dtype="float32") # , mesh=mesh, sharding_rules=SHARDING_RULES)

    input = args.prompt
    # tokens = tokenizer(input, return_tensors="jax")["input_ids"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer(input, return_tensors="jax")["input_ids"]

    with timer("Generating"):
        result = generate(tokens, model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)

    print(tokenizer.decode(result[0]))


def generate(
    input_ids,
    model: Qwen25ForCausalLM,
    max_tokens: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.9,
    rngs = None,
):
    if rngs is None:
        rngs = nnx.Rngs(0)

    batch_size, prompt_len = input_ids.shape
    max_seqlen = prompt_len + max_tokens

    # prefill with full prompt
    outputs = model(input_ids, return_dict=True)
    logits = outputs['logits']
    cache = KVCache.create_from_list(outputs['past_key_values'], model.dtype)

    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    tokens = jnp.concatenate([input_ids, next_token], axis=-1)

    # generate
    for _ in range(max_tokens - 1):
        logits, cache = model.decode(next_token, cache)
        probs = jax.nn.softmax(logits / temperature, axis=-1)[:, 0, :]
        next_token = jnp.argmax(probs, axis=-1)
        next_token = next_token.reshape(1, 1)
        tokens = jnp.concatenate([tokens, next_token], axis=-1)

    return tokens

if __name__ == "__main__":
    main() 