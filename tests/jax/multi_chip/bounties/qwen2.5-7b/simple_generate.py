import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import qwen_nnx
from qwen_nnx.generate import Generator
from qwen_nnx.model import QwenModel
import flax.nnx as nnx

WEIGHTS = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"

tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
model = QwenModel.load_from_hf_pt_model(WEIGHTS, dtype=jnp.bfloat16)

prompt = "[INST] Hello, how are you? [/INST]"
tokens = jnp.array(tokenizer.encode(prompt))
all_tokens = tokens.tolist()

generator = Generator(model, max_seqlen=512)

generated = generator.generate(tokens, nnx.Rngs(0), max_tokens=50, temp=0.7, top_p=0.8)
print(tokenizer.decode(generated)) 