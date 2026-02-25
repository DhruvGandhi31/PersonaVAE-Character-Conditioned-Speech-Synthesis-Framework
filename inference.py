# inference script for Zhongli-TTS (NEED TO MAKE CHANGES TO RUN INFERENCE)
import os
import torch
import soundfile as sf
from src.models import get_model
from src.hparams import hparams as hps
from src.utils import load_checkpoint
from src.txt_clean import cleaned_text_to_sequence
from src.audio import mel_to_audio

 
# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "logs\\zhongli_base\\G_0.pth"  # path to trained generator
output_wav = "output.wav"                  
text_input = "Hello, this is a test of Zhongli-TTS." 

 
# LOAD GENERATOR
print("Loading generator...")
net_g = get_model(hps)
net_g.to(device)
net_g.eval()
load_checkpoint(checkpoint_path, net_g)
print("Generator loaded.")

 
# TEXT TO SEQUENCE
sequence = cleaned_text_to_sequence(text_input)
sequence = torch.LongTensor(sequence).unsqueeze(0).to(device)  # batch size 1

# SYNTHESIZE 
with torch.no_grad():
    # net_g returns: y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
    y_hat, _, _, _, _, _, _ = net_g(sequence)

# Convert mel-spectrogram to waveform
audio = mel_to_audio(y_hat.squeeze(0).cpu().numpy())

# Save audio
sf.write(output_wav, audio, hps.data.sampling_rate)
print(f"Audio saved to {output_wav}")