import torch

from src import utils

file_path = "logs/zhongli_base/G_15000.pth"

model_state_dict = torch.load(file_path)

print(model_state_dict.keys())

#save the state dict to a new txt file
# with open("logs/zhongli_base/G_15000_debug.txt", "w") as f:

#     f.write(f"{model_state_dict['iteration']}\n")