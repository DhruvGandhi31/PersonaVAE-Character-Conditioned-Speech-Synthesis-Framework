#5
import random

input_file = "data/clean_v1/metadata.csv"
train_file = "data/clean_v1/metadata_train.csv"
val_file = "data/clean_v1/metadata_val.csv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

# Split 95% / 5%
split_index = int(len(lines) * 0.95)
train_lines = lines[:split_index]
val_lines = lines[split_index:]

#train
with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_lines)
#val
with open(val_file, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"Total: {len(lines)}")
print(f"Train: {len(train_lines)}")
print(f"Val: {len(val_lines)}")
