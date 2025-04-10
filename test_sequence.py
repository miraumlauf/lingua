from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, AdamW
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import login

# Directly pass your Hugging Face token
HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

repo_name = "umlauf/lingua_3fh_sequence_128_15300"


model = AutoModelForSequenceClassification.from_pretrained(repo_name, trust_remote_code=True)
# print("model:", model)
# tokenizer = AutoTokenizer.from_pretrained(repo_name, legacy=True)
# tokenizer.pad_token = tokenizer.eos_token

# # Prepare input text
# input_text = "The movie was horror!"
# input_dict = tokenizer(input_text, return_tensors="pt")
# print("Input Dict", input_dict)

# Create a small toy dataset
train_inputs = torch.randint(0, model.config.vocab_size, (10, 10))  # 10 samples, 10 tokens each
train_labels = torch.randint(0, 2, (10,))  # Binary classification

# DataLoader
dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Train Loop
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):  # 3 epochs
    total_loss = 0
    for batch in train_loader:
        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



model.eval()  # Set to evaluation mode

test_inputs = torch.randint(0, model.config.vocab_size, (5, 10))  # 5 test samples
test_labels = torch.randint(0, 2, (5,))

with torch.no_grad():  # No gradients needed for inference
    outputs = model(test_inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

accuracy = (predictions == test_labels).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# input_ids = torch.randint(0, model.config.vocab_size, (2, 10))  # Batch size 2, seq length 10
# labels = torch.tensor([1, 0])  # Assume binary classification

# # check model update
# model.train()  # Ensure model is in training mode
# optimizer = AdamW(model.parameters(), lr=5e-5)
# # make copy of old weights and detach
# old_weights = model.classifier.weight.clone().detach()
# # Forward pass
# outputs = model(input_ids, labels=labels)
# loss = outputs.loss

# # Backward pass
# loss.backward()
# optimizer.step()
# # make copy of new weights and detach 
# new_weights = model.classifier.weight.clone().detach()

# if not torch.equal(old_weights, new_weights):
#     print("✅ Model weights updated successfully!") # correct!
# else:
#     print("Warning: Model weights did NOT change!")

# # test loss calculation -> works


# outputs = model(input_ids, labels=labels)

# print("Logits:", outputs.logits)
# print("Loss:", outputs.loss)



# # test inference -> works
# with torch.no_grad():
#     output = model(**input_dict)


# logits = output.logits  # Get class logits
# print("Logits", logits)
# predicted_class = torch.argmax(logits, dim=-1).item()  # Get predicted label

# # Print classification result
# print(f"Predicted class: {predicted_class}")

