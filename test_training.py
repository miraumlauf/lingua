from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AdamW
import torch
from huggingface_hub import login

# Directly pass your Hugging Face token
HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

repo_name = "umlauf/llama_causal_128_15300"


model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo_name, legacy=True)
tokenizer.pad_token = tokenizer.eos_token

batch_size = 1
seq_length = 20
# Change this according to your model config

input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))  # Simulated input
labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))  # Simulated labels


output = model(input_ids, labels=labels)
print("Loss:", output.loss.item() if output.loss is not None else "None")
print("Logits shape:", output.logits.shape) 


