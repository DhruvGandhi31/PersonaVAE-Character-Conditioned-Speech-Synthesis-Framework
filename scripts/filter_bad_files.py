#2
import os
import shutil
import numpy as np
import soundfile as sf

BASE_DIR = "./data"
SOURCE_DIR = os.path.join(BASE_DIR, "zhongli_en")
BAD_DIR = os.path.join(BASE_DIR, "bad_data")

MIN_DURATION = 0.30        # seconds
SILENCE_THRESHOLD = 1e-4  # amplitude threshold

os.makedirs(BAD_DIR, exist_ok=True)


def is_corrupted_or_silent(wav_path):
    try:
        audio, sr = sf.read(wav_path)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) == 0:
            return True, "Zero length"

        duration = len(audio) / sr
        if duration < MIN_DURATION:
            return True, f"Too short ({duration:.2f}s)"

        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < SILENCE_THRESHOLD:
            return True, "Silent / near zero amplitude"

        return False, "OK"

    except Exception as e:
        return True, f"Corrupted ({str(e)})"


def is_empty_or_invalid_text(txt_path):
    if not os.path.exists(txt_path):
        return True, "Missing transcript"

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) == 0:
            return True, "Empty transcript"

        # # Remove ellipsis-only transcripts like "...", "....."
        # cleaned = text.replace(".", "").strip()
        # if len(cleaned) == 0:
        #     return True, "Only ellipsis"

        return False, "OK"

    except Exception as e:
        return True, f"Text read error ({str(e)})"


def move_pair(base_name):
    wav_file = base_name + ".wav"
    txt_file = base_name + ".txt"

    wav_src = os.path.join(SOURCE_DIR, wav_file)
    txt_src = os.path.join(SOURCE_DIR, txt_file)

    wav_dst = os.path.join(BAD_DIR, wav_file)
    txt_dst = os.path.join(BAD_DIR, txt_file)

    if os.path.exists(wav_src):
        shutil.move(wav_src, wav_dst)

    if os.path.exists(txt_src):
        shutil.move(txt_src, txt_dst)


def main():
    total = 0
    rejected = 0

    for file in os.listdir(SOURCE_DIR):
        if file.endswith(".wav"):
            total += 1
            base_name = os.path.splitext(file)[0]

            wav_path = os.path.join(SOURCE_DIR, file)
            txt_path = os.path.join(SOURCE_DIR, base_name + ".txt")

            audio_bad, audio_reason = is_corrupted_or_silent(wav_path)
            text_bad, text_reason = is_empty_or_invalid_text(txt_path)

            if audio_bad or text_bad:
                reason = audio_reason if audio_bad else text_reason
                print(f"[REJECTED] {file} -> {reason}")
                move_pair(base_name)
                rejected += 1
            else:
                print(f"[OK] {file}")

    print("\nFinished.")
    print(f"Total checked: {total}")
    print(f"Rejected: {rejected}")
    print(f"Kept: {total - rejected}")


if __name__ == "__main__":
    main()
