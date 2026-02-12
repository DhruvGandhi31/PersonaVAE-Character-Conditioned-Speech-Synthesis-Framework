#6
import csv
import sys
import os

def convert_csv_to_txt(csv_file_path, txt_file_path):
    """
    Convert a CSV file to a TXT file, preserving the pipe-separated format.
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                for line in csv_file:
                    # Strip any trailing whitespace/newlines and write the line
                    txt_file.write(line.strip() + '\n')
        print(f"Successfully converted {csv_file_path} to {txt_file_path}")
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python csv_to_txt.py <input_csv> <output_txt>")
    #     sys.exit(1)

    BASE_DIR = "data/preprocessed_v1"
    os.makedirs(BASE_DIR, exist_ok=True)

    #train
    csv_file_path = "data/clean_v1/metadata_train.csv"
    txt_file_path = os.path.join(BASE_DIR, "metadata_train.txt")
    convert_csv_to_txt(csv_file_path, txt_file_path)

    #val
    csv_file_path = "data/clean_v1/metadata_val.csv" 
    txt_file_path = os.path.join(BASE_DIR, "metadata_val.txt")
    convert_csv_to_txt(csv_file_path, txt_file_path)
    
