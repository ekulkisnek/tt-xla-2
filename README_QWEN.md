# TT-XLA Qwen2.5-7B Bounty Implementation

This is a development fork of [tenstorrent/tt-xla](https://github.com/tenstorrent/tt-xla) with added Qwen2.5-7B model implementation for tensor parallel inference and GSM8K math evaluation.

## 🎯 **What's Added**

### **Qwen2.5-7B Bounty Implementation**
- **Location**: `tests/jax/multi_chip/bounties/qwen25-7b/`
- **Instruction-tuned model**: `q25_jax_instruct.py`
- **Tensor parallel model**: `qwen25_tp_final.py`
- **GSM8K evaluation**: `gsm8k_simple.py`
- **Base inference**: `simple_inference.py`

### **Key Features**
✅ **Hybrid tensor parallelism** - automatic single/multi-device support  
✅ **Instruction tuning** - chat template support for math problems  
✅ **GSM8K evaluation** - math reasoning benchmark testing  
✅ **Production-ready** - comprehensive validation and documentation  

## 🚀 **Quick Start**

```bash
# Navigate to Qwen bounty implementation
cd tests/jax/multi_chip/bounties/qwen25-7b/

# Run instruction-tuned model
python q25_jax_instruct.py --model_path ../instruct_weights --prompt "What is 2+2?" --max_tokens 20

# Run tensor parallel model
python qwen25_tp_final.py --model_path ../weights --prompt "Hello" --max_tokens 10 --tp 4

# Evaluate on GSM8K
python gsm8k_simple.py --jax_path ../weights --count 5

# Test instruction tuning
python gsm8k_instruct_test.py --model_path ../instruct_weights --count 3
```

## 📁 **Bounty Files Structure**

```
tests/jax/multi_chip/bounties/qwen25-7b/
├── q25_jax_instruct.py              # Instruction-tuned implementation
├── qwen25_tp_final.py               # Tensor parallel implementation  
├── gsm8k_simple.py                  # GSM8K evaluation script
├── simple_inference.py              # Base inference implementation
├── gsm8k_instruct_test.py           # Instruction tuning tests
├── simple_gsm8k_instruct.py         # Simple instruction testing
├── validate_full_model.py           # Model validation
├── FINAL_SUCCESS_REPORT.md          # Complete implementation report
├── TP_IMPLEMENTATION_SUMMARY.md     # Tensor parallel technical details
└── README_GSM8K.md                  # GSM8K evaluation guide
```

## 📋 **Core Files Dependencies**

- **`gsm8k_simple.py`** → imports from `simple_inference.py`
- **`gsm8k_instruct_test.py`** → imports from `q25_jax_instruct.py`
- **`q25_jax_instruct.py`** → self-contained (no local imports)
- **`qwen25_tp_final.py`** → self-contained (no local imports)

## 🏆 **Bounty Completion Status**

**Development Status**: ✅ **Fully Functional**  
**Validation**: 5/5 tests passed  
**Model**: Qwen2.5-7B (7.6B parameters)  
**Implementation**: Instruction-tuned + Tensor Parallel  

## 🔗 **Base Repository**

This fork is based on the latest [tenstorrent/tt-xla](https://github.com/tenstorrent/tt-xla) main branch.

---

**Bounty Target**: Multi-chip tensor parallel Qwen2.5-7B with GSM8K evaluation  
**Location**: `tests/jax/multi_chip/bounties/qwen25-7b/` 