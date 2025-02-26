from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Directly pass your Hugging Face token
HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

repo_name = "umlauf/llama_causal_128_15300"


model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)

print("Config", model.config)
print("Main Name", model.main_input_name)

# with my repo
tokenizer = AutoTokenizer.from_pretrained(repo_name, legacy=True)
input_text = "Lisa is"
input_dict = tokenizer(input_text, return_tensors="pt")
print("Input Dict", input_dict)
input_dict.pop("attention_mask", None) 

# Generate text
# add parameters to generation config??
output = model.generate(
    **input_dict,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0
    )

print(output)

# Decode and print result
print("Generated text:", tokenizer.decode(output[0], skip_special_tokens=True))

# with Llama Rep
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", subfolder="original", legacy=True)

# with local path
# tokenizer_path = "./tokenizers/llama3/original"
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# input_text = "Anna was"

# inputs = tokenizer(input_text, return_tensors="pt")
# print("Input IDs", inputs)

# decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
# print("Decoded text:", decoded_text)
# # maybe inlcude legacy=false to change warning but this might not be compatible with the training of my model

# torch.manual_seed(42)
# #Generate outputs
# outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
