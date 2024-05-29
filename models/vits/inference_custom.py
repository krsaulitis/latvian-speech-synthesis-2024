import csv
import os
import random

import soundfile as sf
import torch

import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
hps = utils.get_hparams()

model_weight_paths = [
    './weights/model-3',
    './weights/model-6',
    './weights/model-8',
]

model_idx = 'model-3'
model_weight_path = './weights/model-3'

speaker_count = hps.data.n_speakers

model = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=speaker_count,
    **hps.model).to(device)


weight_paths = os.listdir(model_weight_path)

with open('./test_concatenated_w_phones.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    rows = list(reader)
    rows = rows[1:]
    random.seed(42)
    random_rows = random.sample(rows, 40)

# Define batch size
batch_size = 4

# Helper function to create batches
def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


with torch.no_grad():
    for weight_path in weight_paths:
        full_weight_path = os.path.join(model_weight_path, weight_path)
        checkpoint = torch.load(full_weight_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        for batch in create_batches(random_rows, batch_size):
            texts = [row[3] for row in batch]
            files = [row[2] for row in batch]
            tokens_list = [torch.LongTensor(text_to_sequence(text, ['latvian_cleaner_1'], symbols)) for text in texts]

            max_len = max(tokens.size(0) for tokens in tokens_list)
            tokens_padded = torch.zeros(len(tokens_list), max_len).long()
            for i, tokens in enumerate(tokens_list):
                tokens_padded[i, :tokens.size(0)] = tokens

            tokens_padded = tokens_padded.to(device)
            lengths = torch.LongTensor([tokens.size(0) for tokens in tokens_list]).to(device)

            for speaker_id in range(1, speaker_count):
                if not os.path.exists(f'./output/{model_idx}/{speaker_id}'):
                    os.makedirs(f'./output/{model_idx}/{speaker_id}')

                speaker_tensor = torch.LongTensor([speaker_id] * len(tokens_list)).to(device)

                y_hat, attn, mask, *_ = model.infer(tokens_padded, lengths, speaker_tensor)

                y_hat = y_hat.cpu().detach().numpy()
                for i, audio in enumerate(y_hat):
                    audio_path = os.path.join('./output', model_idx, str(speaker_id))
                    if not os.path.exists(audio_path):
                        os.makedirs(audio_path)
                    sf.write(os.path.join(audio_path, files[i]), audio.squeeze(), hps.data.sampling_rate)

    # for i, row in enumerate(random_rows):
    #     text = row[3]
    #     file_name = row[2]
    #
    #     for speaker_id in range(1, speaker_count):
    #         if not os.path.exists(f'./output/{model_idx}/{speaker_id}'):
    #             os.makedirs(f'./output/{model_idx}/{speaker_id}')
    #
    #         tokens = torch.LongTensor([text_to_sequence(text, ['latvian_cleaner_1'], symbols)]).to(device)
    #         speaker_id = torch.LongTensor([speaker_id]).to(device)
    #
    #         y_hat, attn, mask, *_ = model.infer(tokens, torch.LongTensor([tokens.size(1)]).to(device), speaker_id)
    #
    #         audio = y_hat.squeeze().cpu().detach().numpy()
    #         sf.write(f'./output/{model_idx}/{speaker_id}/{file_name}', audio, hps.data.sampling_rate)


# path = f'./weights/G_220000.pth'
# checkpoint = torch.load(path, map_location=device)
# model.load_state_dict(checkpoint['model'])
#
# text = "Es esmu teksta sintēzes modelis, kas spēj runāt latviešu valodā."
# tokens = torch.LongTensor([text_to_sequence(text, ['latvian_cleaner_1'])])
# speaker = torch.LongTensor([100])
#
# y_hat, attn, mask, *_ = model.infer(tokens, torch.LongTensor([tokens.size(1)]), speaker)
#
# audio = y_hat.squeeze().cpu().detach().numpy()
# import soundfile as sf
#
# sf.write('output_220_100.wav', audio, hps.data.sampling_rate)
