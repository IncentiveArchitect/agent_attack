import torch
import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

gc.collect()
torch.cuda.empty_cache()

mirror_name = "NousResearch/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading HF model...")
try:
    hf_model = AutoModelForCausalLM.from_pretrained(
        mirror_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("HF Model loaded successfully!")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(hf_model)
except Exception as e:
    print(f"Failed to load HF model: {e}")
