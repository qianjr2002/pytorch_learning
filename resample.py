import librosa
import soundfile as sf
import os
import numpy as np
import argparse
from tqdm import tqdm

# Set sample rate before and after resampling
original_sr = 48000
target_sr = 16000

# Set the directory of folder of wav files to be resampled and get the list of these wav files
parser = argparse.ArgumentParser()
parser.add_argument("--wav_folder", type=str, help="The wav folder in which the wav files are to be resampled")

args = parser.parse_args()
wav_folder = args.wav_folder

wav_list = os.listdir(wav_folder)

# Create the folder of resampled files
# Change the output directory to be at the same level as the input directory
wav_folder_name = os.path.basename(os.path.normpath(wav_folder))
output_folder = os.path.join(os.path.dirname(wav_folder), wav_folder_name + '_16k')
os.makedirs(output_folder, exist_ok=True)

print(f"Input folder: {wav_folder}")
print(f"Output folder: {output_folder}")

# Start resampling
for wav in tqdm(wav_list, desc="Resampling"):
    # READ + RESAMPLE by librosa
    wav_path = os.path.join(wav_folder, wav)
    wav_16k, _ = librosa.load(wav_path, sr=target_sr)

    # WRITE by soundfile
    output_path = os.path.join(output_folder, wav)
    sf.write(output_path, wav_16k, target_sr)

print(f"Resampled files are saved in {output_folder}")
