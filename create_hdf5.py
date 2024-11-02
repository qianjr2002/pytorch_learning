import os
import h5py
import numpy as np
from scipy.io import wavfile

def create_hdf5_from_wav(wav_dir, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for root, _, files in os.walk(wav_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    sample_rate, data = wavfile.read(file_path)
                    
                    # Store data in HDF5 file
                    dataset_name = os.path.relpath(file_path, wav_dir)
                    dataset_name = dataset_name.replace(os.sep, '_')
                    hdf5_file.create_dataset(dataset_name, data=data)
                    hdf5_file[dataset_name].attrs['sample_rate'] = sample_rate
                    print(f"Added {file_path} to HDF5 file")

noisy_wav_dir = '/home/aq/dataset/noisy_testset_wav'
clean_wav_dir = '/home/aq/dataset/clean_testset_wav'

noisy_hdf5_path = '/home/aq/dataset/noisy_testset.hdf5'
clean_hdf5_path = '/home/aq/dataset/clean_testset.hdf5'

create_hdf5_from_wav(noisy_wav_dir, noisy_hdf5_path)
create_hdf5_from_wav(clean_wav_dir, clean_hdf5_path)

