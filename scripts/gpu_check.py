import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(torch.version.cuda)

# If you want to check for specific GPU details, you can use this code.