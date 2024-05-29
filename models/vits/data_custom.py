import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from mel_processing import spectrogram_torch
from text import text_to_sequence, cleaned_text_to_sequence
from utils import load_wav_to_torch

from text.symbols import symbols
from text.symbols_lv import symbols as symbols_lv


class CustomDatasetV1(Dataset):
    def __init__(self, path, hparams):
        super().__init__()

        df = pd.read_csv(path, delimiter='\t')
        df['path'] = df['path'].apply(lambda x: os.path.join('data/common_voice/', x))
        self.paths_texts = df.values.tolist()
        self.client_ids = df['client_id'].unique()

        # Hyperparameters
        self.cleaned_text = hparams.cleaned_text
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.use_phonemes = hparams.use_phonemes

        # random.seed(1234)
        # random.shuffle(self.paths_texts)

    def get_audio_text_speaker_pair(self, index):
        path_text = self.paths_texts[index]
        spec, wav = self.get_audio_speaker(path_text[1])
        sid = self.get_sid(path_text[0])
        raw_text = path_text[21] if self.use_phonemes else path_text[3]
        text = self.get_text(raw_text)

        return text, spec, wav, sid, raw_text

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners, symbols_lv if self.use_phonemes else symbols)

        return torch.LongTensor(text_norm)

    def get_audio_speaker(self, path):
        audio, sampling_rate = load_wav_to_torch(path, self.sampling_rate, True)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))

        # audio_norm = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        spec_filename = path.replace(".wav", ".spec.pt")

        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        return spec, audio_norm

    def get_sid(self, sid):
        # return torch.LongTensor([1])
        sid = self.client_ids.tolist().index(sid) + 1
        return torch.LongTensor([int(sid)])

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(index)

    def __len__(self):
        return len(self.paths_texts)
