import pandas as pd
import soundfile as sf
import torch

import commons
import utils
from mel_processing import mel_spectrogram_torch
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols
from utils import load_wav_to_torch

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
hps = utils.get_hparams()

model = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)

checkpoint = torch.load('./weights/G_300015.pth', map_location=device)
model.load_state_dict(checkpoint['model'])

df = pd.read_csv('./data/test.tsv', sep='\t')
first_row = df.iloc[0]

text = first_row['sentence']
path = first_row['path']

audio, sampling_rate = load_wav_to_torch(
    f'./data/common_voice/{path}',
    hps.data.sampling_rate,
    True,
)

y = audio.unsqueeze(0).unsqueeze(0).to(device)
y_lengths = torch.LongTensor([y.size(2)])

tokens = torch.LongTensor([text_to_sequence(text, ['latvian_cleaner_1'], symbols)])
speaker = torch.LongTensor([1])

y_hat, attn, mask, *_ = model.infer(tokens, torch.LongTensor([tokens.size(1)]), speaker)

y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

y_hat_sliced, ids_slice = commons.rand_slice_segments(y_hat, y_hat_lengths, hps.train.segment_size, y_lengths)
y_sliced = commons.slice_segments(y, ids_slice, hps.train.segment_size)

y_mel = mel_spectrogram_torch(
    y_sliced.squeeze(1),
    hps.data.filter_length,
    hps.data.n_mel_channels,
    hps.data.sampling_rate,
    hps.data.hop_length,
    hps.data.win_length,
    hps.data.mel_fmin,
    hps.data.mel_fmax
)
y_hat_mel = mel_spectrogram_torch(
    y_hat_sliced.squeeze(1),
    hps.data.filter_length,
    hps.data.n_mel_channels,
    hps.data.sampling_rate,
    hps.data.hop_length,
    hps.data.win_length,
    hps.data.mel_fmin,
    hps.data.mel_fmax
)

mel_loss = torch.nn.functional.l1_loss(y_mel, y_hat_mel)

print(f'Mel loss: {mel_loss.item()}')

# sf.write(f'./original.wav', y.squeeze().cpu().detach().numpy(), hps.data.sampling_rate)
# sf.write(f'./generated.wav', y_hat.squeeze().cpu().detach().numpy(), hps.data.sampling_rate)

sf.write(f'./original.wav', y_sliced.squeeze().cpu().detach().numpy(), hps.data.sampling_rate)
sf.write(f'./generated.wav', y_hat_sliced.squeeze().cpu().detach().numpy(), hps.data.sampling_rate)

