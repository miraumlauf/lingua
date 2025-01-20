from transformers import LlamaForCausalLM, AutoTokenizer
import torch


repo_name = "umlauf/llama_lr"

model = LlamaForCausalLM.from_pretrained(repo_name)
print("Config", model.config)
# with my repo
tokenizer = AutoTokenizer.from_pretrained(repo_name, legacy=True)

# with Llama Rep
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", subfolder="original", legacy=True)

# with local path
# tokenizer_path = "./tokenizers/llama3/original"
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

input_text = "Anna was"

inputs = tokenizer(input_text, return_tensors="pt")
print("Input IDs", inputs)

decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
print("Decoded text:", decoded_text)
# maybe inlcude legacy=false to change warning but this might not be compatible with the training of my model

torch.manual_seed(42)
#Generate outputs
outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
