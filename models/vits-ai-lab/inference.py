import os
import sys
import csv
import phonemizer.phonemize
import torch
import scipy
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cvutils import Phonemiser
from phonemizer import phonemize

# from config import symbols

sys.path.append('./model')
import utils as utils
import commons as commons
from models import SynthesizerTrn
from text.symbols import symbols
from text.symbols_lv import symbols as symbols_lv
from text import text_to_sequence

# phonemizer.backend.
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'


class VITSClarinLV:
    def __init__(self, mode='lv'):
        self.used_symbols = symbols_lv if mode == 'lv' else symbols
        self.used_config = './model/configs/latvian.json' if mode == 'lv' else './model/configs/ljs_base.json'
        self.used_checkpoint = './model/checkpoints/VITS_Latvian_G.pth' if mode == 'lv' else './model/checkpoints/pretrained_ljs.pth'

        self.hps = utils.get_hparams_from_file(self.used_config)

        self.model = SynthesizerTrn(
            len(self.used_symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )

        self.model.eval()

        utils.load_checkpoint(self.used_checkpoint, self.model, None)

    def generate(self, text, path):
        with torch.no_grad():
            tokens = self.get_text(text)
            t_in = tokens.unsqueeze(0)
            t_in_len = torch.LongTensor([tokens.size(0)])
            sid = torch.LongTensor([4])
            audio = self.model.infer(
                t_in,
                t_in_len,
                noise_scale=.667,
                noise_scale_w=0.8,
                length_scale=1
            )[0][0, 0]

            # Normalize to int16 for 16bit audio file
            resampled = librosa.resample(audio.numpy(), orig_sr=self.hps.data.sampling_rate, target_sr=16000)
            waveform = (resampled / np.max(np.abs(resampled)) * 32767).astype(np.int16)
            scipy.io.wavfile.write(path, rate=16000, data=waveform)

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners, self.used_symbols)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


def read_phoneme_map():
    """Reads a TSV file with phoneme mappings and returns a dictionary."""
    phoneme_map = {}
    with open('./config/phoneme_map.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                phoneme_map[parts[0]] = parts[1]
    return phoneme_map


def map_lv_text(text):
    def map_text(text, phoneme_map):
        words = text.split(' ')
        mapped_words = []

        for word in words:
            # Split phonemes within the word, remove underscores
            # phonemes = word
            # mapped_phonemes = []
            # skip = False

            # for i, phoneme in enumerate(phonemes):
            #     if skip:
            #         skip = False
            #         continue
            #
            #     diphones = phonemes[i:i + 2]
            #     mapped = phoneme_map.get(diphones)
            #     if mapped:
            #         skip = True
            #     else:
            #         mapped = phoneme_map.get(phoneme, phoneme)
            #     mapped_phonemes.append(mapped)

            phonemes = word.split('_')
            mapped_phonemes = [phoneme_map.get(phoneme, phoneme) for phoneme in phonemes]
            # Join the mapped phonemes to form the word back, without underscores
            mapped_word = ''.join(mapped_phonemes)
            mapped_words.append(mapped_word)

        # Join the mapped words with spaces to maintain word separation
        return ' '.join(mapped_words)

    phoneme_map = read_phoneme_map()
    return map_text(text, phoneme_map)


def phonemise_lv(text):
    p = Phonemiser('lv')
    phoneme_map = read_phoneme_map()
    words = text.split(' ')
    phonemised_words = []

    for word in words:
        phonemes = p.phonemise(word)
        # mapped_phonemes = [phoneme_map.get(phoneme, phoneme) for phoneme in phonemes]
        # phonemised_word = ''.join(mapped_phonemes)

        # remove '̪' symbol

        # phonemes = phonemes.replace('̪', '').replace('̯̯', '').replace('͡', '').replace('ʒ', 'ʓ').replace('dz', 'ʣ')

        phonemised_words.append(phonemes)

    return ' '.join(phonemised_words)


# text_lv_raw = 'šis ir ļoti īpašs šaursliežu dzelzsceļš'
# text_lv_raw = 'varu pat aizmigt'
#
# model = VITSClarinLV('lv')
# text_lv_phonemes = 'ʃis ir ʎɔtĭ iːpɑʃ ʃɑ͜ursli͜eʓŭ ʣelsʦeʎʃ'
# # text_lv_phonemes_underscore = 'ʃ_i_s i_r ʎ_ɔ_t_ĭ iː_p_ɑ_ʃ ʃ_ɑ͜u_r_s_l_i͜e_ʓ_ŭ ʣ_e_l_s_s_ʦ_e_ʎ_ʃ'
# text_lv = map_lv_text(text_lv_phonemes)
#
# p = Phonemiser('lv')
# text_lv_alt_phonemes = phonemise_lv(text_lv_raw)
# text_lv_alt_2_phonemes = phonemize(text_lv_raw, language='lv', backend='espeak', with_stress=True)
# text_lv_alt = map_lv_text(text_lv_alt_phonemes)
#
# model.generate(text_lv_alt, 'test_lv_alt.wav')


model = VITSClarinLV('lv')
# model.generate('This is a test example', 'test_en.wav')

def generate():
    with open('./test_concatenated_w_phones.tsv', 'r') as file:
        rows = csv.reader(file, delimiter='\t')

        for i, row in tqdm(enumerate(rows, start=0), total=100):
            if i == 0:
                continue
            # if i > 10000:
            #     break

            sentence = map_lv_text(row[4])
            file = row[2]
            path = f'./audios_test/{file}'
            model.generate(sentence, path)


generate()
