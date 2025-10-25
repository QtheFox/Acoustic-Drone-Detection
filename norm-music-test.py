# imports
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import stft
from random import uniform, sample
from pyroomacoustics import doa, Room, ShoeBox
import time


# ... your code here ...

from scipy.io import wavfile

# Replace 'filename.wav' with your WAV file path
sample_rate, audio_data = wavfile.read('data/examples/DRONE_001_L.wav')

print("Sample rate:", sample_rate)
print("Audio data shape:", audio_data.shape)

audio_data=audio_data[0:sample_rate*3]
audio_float = audio_data.astype(np.float32)
# Normalize to [-1, 1]
audio_normalized = audio_float / np.max(np.abs(audio_float))
# To convert back to int16 after processing:
audio_int16 = (audio_normalized * 32767).astype(np.int16)

signal=audio_int16

# constants / config
fs = sample_rate
nfft = 1024
n = len(audio_data) # simulation length of source signal (3 seconds)
#signal=np.random.random(n)
n_frames = 30
max_order = 10
doas_deg = np.linspace(start=0, stop=359, num=360, endpoint=True)
doa_deg = 170
rs = [0.5, 1, 1.5]
mic_center = np.c_[[5,5,1]]
mic_locs = mic_center + np.c_[[ 0.0,  0.0, 0.15],
                              [ 0.15,  0.0, 0.0],
                              [-0.15,  0.0, 0.0],
                              [ 0.0, 0.15, 0.0],


]
snr_lb, snr_ub = 0, 30



# room simulation
data = []


doa_rad = np.deg2rad(doa_deg)
source_loc = mic_center[:,0] + np.c_[1*np.cos(doa_rad), 1*np.sin(doa_rad), 0][0]
room_dim = [10, 10, 5] # meters

room = ShoeBox(room_dim, fs=fs, max_order=max_order)
room.add_source(source_loc, signal)
room.add_microphone_array(mic_locs)
room.simulate(snr=uniform(snr_lb, snr_ub))
signals = room.mic_array.signals

# calculate n_frames stft frames starting at 1 second
stft_signals = stft(signals[:,fs:fs+n_frames*nfft], fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
data.append([1, doa_deg, stft_signals])
start = time.perf_counter()

#prediction
kwargs = {'L': mic_locs,
          'fs': fs, 
          'nfft': nfft,
          'num_src':1,
          'azimuth': np.deg2rad(np.arange(360)),
          #'colatitude': np.pi/2*np.ones(1),
           'dim': 2                   
                                
}
algorithms = {
    'MUSIC': doa.music.MUSIC(**kwargs),
    'NormMUSIC': doa.normmusic.NormMUSIC(**kwargs),
    'NormMUSIC': doa.normmusic.NormMUSIC(**kwargs),
    'SRP-PHAT': doa.srp.SRP(**kwargs),
    'CSSM': doa.cssm.CSSM(**kwargs),
    'WAVES':doa.waves.WAVES(**kwargs),
    'TOPS':doa.tops.TOPS(**kwargs),
    'FRIDA':doa.frida.FRIDA(**kwargs)
}

columns = ["r", "DOA"] + list(algorithms.keys())

predictions = {n:[] for n in columns}
for r, doa_deg, stft_signals in data:
    predictions['r'].append(r)
    predictions['DOA'].append(doa_deg)
    for algo_name, algo in algorithms.items():
        algo.locate_sources(stft_signals)
        #print(np.rad2deg(algo.colatitude_recon[0]))
        print(np.rad2deg(algo.azimuth_recon[0]))#,np.rad2deg(algo.azimuth_recon[1]),np.rad2deg(algo.azimuth_recon[2]),np.rad2deg(algo.azimuth_recon[3]))
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.4f} seconds")
        predictions[algo_name].append(np.rad2deg(algo.azimuth_recon[0]))
df = pd.DataFrame.from_dict(predictions)
print(df)


