"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


# MODEL = "bigram"  # bigram or minigpt
MODEL = "minigpt"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
if config.to_log:
    wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model.to(device)

best_train_loss = float('inf')  # Initialize best training loss to infinity

validate_every_n_batches = 10

for epoch in range(1):
    train_loss = 0
    model.train()  
    batch_count = 0

    for iteration, (inputs, next_words) in enumerate(train_dataloader, 1):
        inputs, next_words = inputs.to(device), next_words.to(device)

        optim.zero_grad()
        logits = model(inputs)
        logits = rearrange(logits, 'b n d -> (b n) d')
        next_words = rearrange(next_words, 'b n -> (b n)')
        loss = criterion(logits, next_words)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        batch_count += 1

        if iteration % config.log_interval == 0:
            wandb.log({"Training Loss": loss.item(), "Epoch": epoch, "Iteration": iteration})

        if batch_count % validate_every_n_batches == 0:
            model.eval() 
            val_loss = 0
            num_samples_processed = 0
            max_samples = 1000

            with torch.no_grad():  
                for eval_iteration, (eval_inputs, eval_next_words) in enumerate(eval_dataloader):
                    eval_inputs, eval_next_words = eval_inputs.to(device), eval_next_words.to(device)
                    
                    logits = model(eval_inputs)
                    logits = rearrange(logits, 'b n d -> (b n) d')
                    eval_next_words = rearrange(eval_next_words, 'b n -> (b n)')
                    loss = criterion(logits, eval_next_words)
                    val_loss += loss.item()
                    if eval_iteration % config.log_interval == 0:
                        wandb.log({"Val Loss": loss.item(), "Epoch": epoch, "Eval Iteration": eval_iteration})
                
                    num_samples_processed += len(eval_inputs)
                    if num_samples_processed >= max_samples:
                        break
                        
            model.train() 

    avg_train_loss = train_loss / len(train_dataloader)

    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        model_path = f"/best_models/best_model_epoch_{epoch}_train_loss_{best_train_loss:.4f}.pt"
        torch.save(model.state_dict(), model_path)

### ==================== END OF YOUR CODE ==================== ###



