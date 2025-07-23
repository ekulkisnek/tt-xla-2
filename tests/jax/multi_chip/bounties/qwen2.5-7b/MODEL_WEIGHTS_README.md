# 📥 Model Weights Setup

## Overview

The Qwen2.5-7B model weights are not included in this repository due to their large size (~15GB). You need to download them separately.

## Download Instructions

### Option 1: HuggingFace Hub (Recommended)

```bash
# Navigate to the qwen2.5-7b directory
cd tests/jax/multi_chip/bounties/qwen2.5-7b

# Download from HuggingFace Hub
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct qwen25_7b_instruct_weights
```

### Option 2: Manual Download

1. Visit [Qwen2.5-7B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
2. Download the following files:
   - `model-00001-of-00004.safetensors`
   - `model-00002-of-00004.safetensors`
   - `model-00003-of-00004.safetensors`
   - `model-00004-of-00004.safetensors`
   - `model.safetensors.index.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `config.json`
3. Place them in the `qwen25_7b_instruct_weights/` directory

## Directory Structure

After downloading, your directory should look like:

```
qwen2.5-7b/
├── qwen_nnx/                    # Model implementation
├── qwen25_tp_final.py          # Tensor parallel code
├── qwen25_7b_instruct_weights/ # Model weights (download required)
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── model.safetensors.index.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── config.json
└── MODEL_WEIGHTS_README.md     # This file
```

## Verification

After downloading, you can verify the setup by running:

```bash
# Check file sizes (should be ~15GB total)
ls -lh qwen25_7b_instruct_weights/*.safetensors

# Test basic loading
python -c "
import jax.numpy as jnp
from transformers import AutoTokenizer
import qwen_nnx

tokenizer = AutoTokenizer.from_pretrained('qwen25_7b_instruct_weights')
model = qwen_nnx.QwenModel.load_from_hf_pt_model('qwen25_7b_instruct_weights', dtype=jnp.bfloat16)
print('✅ Model loaded successfully!')
"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: The model requires ~15GB of RAM/VRAM
2. **Download Timeout**: Use `git lfs` for reliable downloads
3. **Permission Errors**: Ensure write permissions in the directory

### Alternative Models

If you need a smaller model for testing:
- Consider using Qwen2.5-1.5B or Qwen2.5-3B
- Update the model path in your code accordingly

## License

The model weights follow the original Qwen2.5-7B license. Please review the license terms on the HuggingFace page before use. 