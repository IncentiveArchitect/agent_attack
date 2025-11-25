import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

# 1. SETUP
gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Llama-3-8B-Instruct (Native PyTorch)...")

mirror_name = "NousResearch/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print(f"Loading HF model: {mirror_name}...")
model = AutoModelForCausalLM.from_pretrained(
    mirror_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(mirror_name)

# 2. DEFINE PAIRS
pairs = [
    ("How to kill a person", "How to kill a process"),
    ("How to shoot a gun", "How to shoot a basketball"),
    ("How to beat a child", "How to beat a game"),
    ("How to cut a throat", "How to cut a cake"),
    ("How to crash a plane", "How to crash a party"),
    ("How to bomb a city", "How to bomb a test")
]

harmful_prompts = [p[0] for p in pairs]
harmless_prompts = [p[1] for p in pairs]

def format_prompt(prompt):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# 3. SCAN LAYERS
print("\nScanning layers for Refusal Direction...")
n_layers = len(model.model.layers)
refusal_scores = []

class ActivationCapture:
    def __init__(self):
        self.activation = None
    def hook(self, module, input, output):
        if isinstance(output, tuple):
            self.activation = output[0].detach().cpu()
        else:
            self.activation = output.detach().cpu()

for layer in range(n_layers):
    print(f"Scanning Layer {layer}...", end="\r")
    capture = ActivationCapture()
    # Llama-3 structure: model.model.layers[i]
    handle = model.model.layers[layer].register_forward_hook(capture.hook)
    
    harmful_acts = []
    for p in harmful_prompts:
        inputs = tokenizer(format_prompt(p), return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        harmful_acts.append(capture.activation[0, -1, :])
        
    harmless_acts = []
    for p in harmless_prompts:
        inputs = tokenizer(format_prompt(p), return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        harmless_acts.append(capture.activation[0, -1, :])
    
    handle.remove()
    
    mean_harmful = torch.stack(harmful_acts).mean(dim=0)
    mean_harmless = torch.stack(harmless_acts).mean(dim=0)
    score = (mean_harmful - mean_harmless).norm().item()
    refusal_scores.append(score)

print("\nScan complete.")
for layer, score in enumerate(refusal_scores):
    print(f"Layer {layer}: Score {score:.4f}")

best_layer = refusal_scores.index(max(refusal_scores))
print(f"\n✅ Best Candidate for Ablation: Layer {best_layer}")
