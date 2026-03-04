# inference script for Zhongli-TTS

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

import torch
from src import utils
from src import commons
import soundfile as sf

from src.models import SynthesizerTrn
from src.txt_clean.symbols import symbols
from src.txt_clean import text_to_sequence
from src.txt_clean import _clean_text  # <-- import internal cleaner for debugging


# CONFIG
config_path = "configs/zhongli_base.json"
checkpoint_path = "logs/zhongli_base/G_55000.pth"
text_input = "Hello, how are you? It is indeed a pleasure to meet you. I hope you have a wonderful day!" 
output_wav = "zhongli_output.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TEXT PROCESSING
def get_text(text, hps):
    print("\n===== TEXT DEBUG =====")
    print("Original text:", text)

    # Step 1: Clean + phonemize (what phonemizer outputs)
    cleaned = _clean_text(text, hps.data.text_cleaners)
    print("After cleaners / phonemizer:", cleaned)

    # Step 2: Convert to symbol IDs
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    print("Token IDs:", text_norm)
    print("Token length:", len(text_norm))

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)

    print("======================\n")

    return torch.LongTensor(text_norm)


# LOAD CONFIG + MODEL
print("Loading config...")
hps = utils.get_hparams_from_file(config_path)

print("Building model...")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=0,
    **hps.model
).to(device)

_ = net_g.eval()

print("Loading checkpoint...")
_ = utils.load_checkpoint(checkpoint_path, net_g, None)

print("Model ready.")


# INFERENCE
stn_tst = get_text(text_input, hps)
x_tst = stn_tst.unsqueeze(0).to(device)
x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

with torch.no_grad():
    audio = net_g.infer(
        x_tst,
        x_tst_lengths,
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0
    )[0][0, 0].data.cpu().float().numpy()


# SAVE OUTPUT
sf.write(output_wav, audio, hps.data.sampling_rate)
print(f"Saved audio to {output_wav}")