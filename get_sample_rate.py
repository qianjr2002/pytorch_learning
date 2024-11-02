import sys
import librosa

def get_sample_rate(audio_path):
    try:
        # Load the audio file
        _, sr = librosa.load(audio_path, sr=None)
        print(f"The sample rate of the audio file is: {sr} Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_sample_rate.py <audio_file_path>")
    else:
        audio_path = sys.argv[1]
        get_sample_rate(audio_path)
'''
python get_sample_rate.py ../dataset/noisy_testset_wav/p232_001.wav
'''
