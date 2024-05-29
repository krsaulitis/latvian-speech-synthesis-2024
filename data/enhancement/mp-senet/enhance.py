import json
import os

import librosa
import soundfile as sf
import torch

from datasets.dataset import mag_pha_stft, mag_pha_istft
from env import AttrDict
from models.generator import MPNet

config_file = './best_ckpt/config.json'
with open(config_file) as f:
    data = f.read()

global h

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

json_config = json.loads(data)
h = AttrDict(json_config)
model = MPNet(h).to(device)

state_dict = torch.load('best_ckpt/g_best', map_location=device)
model.load_state_dict(state_dict['generator'])


def enhance(file_path, output_path):
    noisy_wav, sr = librosa.load(file_path, sr=h.sampling_rate)
    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    audio_g = audio_g / norm_factor

    sf.write(output_path, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


model.eval()
with torch.no_grad():
    file_paths = '../files/test_non_enhanced'

    for file_path in os.listdir(file_paths):
        if os.path.exists(f"./test-mp-senet/{file_path}"):
            print(f"Skipping {file_path}")
            continue

        if file_path.endswith(".wav"):
            enhance(f"{file_paths}/{file_path}", f"./test-mp-senet/{file_path}")



