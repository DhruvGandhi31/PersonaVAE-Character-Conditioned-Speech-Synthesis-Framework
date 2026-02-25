# PersonaVAE: Character-Conditioned Speech Synthesis Framework

PersonaVAE is an educational project that explores voice synthesis by simulating the voice of **Zhongli**, a character from the game *Genshin Impact*, using a **Conditional Variational Autoencoder (CVAE)** architecture.

The goal of this project is to experiment with deep learning–based speech generation techniques and demonstrate how character-specific voice modeling can be approached in a research context.

---

## ⚠️ Important Notice

1. **Educational Use Only:**
   This project is intended strictly for educational and research purposes. It is **NOT** for commercial use.

2. **Copyright Disclaimer:**
   All rights to the character Zhongli belong to **miHoYo / HoYoverse**.
   The original English voice performance is by **Keith Silverstein**.

3. **Commercial Restrictions:**
   Do not use this project, its outputs, or any derived materials for commercial purposes without prior written permission from the respective rights holders.

4. **No Voice Data or Model Weights Provided:**
   Due to copyright restrictions, extracted voice data and trained model weights are **not included** in this repository.

5. **Liability Disclaimer:**
   Users are solely responsible for how they use this project and any generated content. The repository maintainer assumes no responsibility for misuse.

---


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak` (Linux)
       (For Windows please refer this: [Espeak downloads](https://espeak.sourceforge.net/download.html))
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd src/monotonic_align/
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training
```sh
python src/train.py -c configs/zhongli_base.json -m zhongli_base
```

P.S If you have the training files you can keep them under the dataset folder and update the config accordingly.
---

## Acknowledgement

This project references publicly available datasets, including:

* [Genshin-Voice Dataset](https://huggingface.co/datasets/simon3000/genshin-voice)


