import os

import mutagen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydub import AudioSegment, effects
from pydub.silence import detect_silence, detect_nonsilent, split_on_silence


def recalculate_lengths():
    base_path = './enhanced-missing/enhanced/'
    data = pd.read_csv('./enhanced-missing/missing.tsv', sep='\t')
    total_length = 0

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        full_path = os.path.join(base_path, row['path'].replace('.mp3', '.wav'))
        audio = mutagen.File(full_path, easy=True)
        length = audio.info.length
        data.at[idx, 'audio_length'] = length
        total_length += length

    print(f"Total length: {total_length / 60} minutes, {total_length / 3600} hours")
    data.to_csv('./enhanced-missing/missing_re.tsv', sep='\t', index=False)


# recalculate_lengths()
# exit()


def create_train_test_split():
    data = pd.read_csv('./data_for_adobe_re_w_missing.tsv', sep='\t')
    print(len(data))

    data['path'] = data['path'].str.replace('.mp3', '.wav')

    grouped = data.groupby('client_id')

    test_split = pd.DataFrame()

    for name, group in grouped:
        group['cumulative_audio_length_minutes'] = group['audio_length'].cumsum() / 60
        total_length = group['audio_length'].sum() / 60
        if total_length < 10:
            print(f"Skipping {name} with total length of {total_length} minutes")
            data.drop(group.index, inplace=True)
            continue

        has_over_30 = False
        for idx, row in group.iterrows():
            if row['cumulative_audio_length_minutes'] > 30:
                test_split = test_split._append(row, ignore_index=True)
                data.drop(idx, inplace=True)
                has_over_30 = True

        if not has_over_30:
            group_test_split = group.sample(1)
            test_split = pd.concat([test_split, group_test_split])
            data.drop(group_test_split.index, inplace=True)

    print(len(test_split))
    print(len(data))

    print(f"Test split length: {test_split['audio_length'].sum() / 60} minutes or {test_split['audio_length'].sum() / 3600} hours")
    print(f"Train split length: {data['audio_length'].sum() / 60} minutes or {data['audio_length'].sum() / 3600} hours")

    print(f"Test split clients: {len(test_split['client_id'].unique())}")
    print(f"Train split clients: {len(data['client_id'].unique())}")

    data.to_csv('./train.tsv', sep='\t', index=False)
    test_split.to_csv('./test.tsv', sep='\t', index=False)


# create_train_test_split()
# exit()


def deconcatenate():
    data = pd.read_csv('./enhanced-full/8b4d10.tsv', sep='\t')
    grouped = data.groupby('client_id')

    for name, group in grouped:
        file_path = f'./enhanced-full/{name}_full-enhanced-90p.wav'
        audio = AudioSegment.from_file(file_path)

        current_time = 0
        for index, row in group.iterrows():
            cumulative_len = group['cumulative_audio_length_minutes'][index]
            if cumulative_len > 60:
                break

            # if (index) % 2 != 0 or index == len(group) - 1:
            #     continue

            def get_file_without_extension(path):
                file = path.split('/')[-1]
                return file.split('.')[0]

            file_without_extension = get_file_without_extension(row['path'])
            # next_file_without_extension = get_file_without_extension(group.iloc[index + 1]['path'])

            duration = row['audio_length']
            # next_duration = group.iloc[index + 1]['audio_length']

            def trim_silence(segment):
                margin = 200
                non_silence = detect_nonsilent(segment, min_silence_len=100, silence_thresh=segment.dBFS-16)
                trim_start = non_silence[0][0] - margin if non_silence[0][0] > margin else 0
                trim_end = non_silence[-1][1] + margin if non_silence[-1][1] + margin < len(segment) else len(segment)
                return segment[trim_start:trim_end]

            segment = audio[current_time:current_time + duration]
            current_time += duration + 300
            trimmed_segment = trim_silence(segment)

            # next_segment = audio[current_time:current_time + next_duration]
            # current_time += next_duration + 300
            # next_trimmed_segment = trim_silence(next_segment)

            combined = trimmed_segment #+ next_trimmed_segment

            os.makedirs(f'./enhanced-full/{name}-detailed', exist_ok=True)
            # combined.export(f'./enhanced-full/{name}-full/{file_without_extension}_{next_file_without_extension}.wav', format='wav')
            combined.export(f'./enhanced-full/{name}-detailed/{file_without_extension}.wav', format='wav')

        break


deconcatenate()
exit()


def concatenate():
    data = pd.read_csv('./enhanced-full/8b4d10.tsv', sep='\t')
    grouped = data.groupby('client_id')
    client_count = len(grouped)

    for name, group in grouped:
        client_count -= 1
        print(f"Processing {name} ({client_count} clients left)")
        folder_path = f'../../sets/common_voice/clips/{name}'

        silence = AudioSegment.silent(duration=300)
        combined = AudioSegment.silent(duration=0)

        for idx, file_path in tqdm(enumerate(group['path']), desc=f"Processing {name}"):
            cumulative_len = group['cumulative_audio_length_minutes'][idx]
            if cumulative_len < 120 or cumulative_len > 180:
                continue

            file = file_path.split('/')[-1]
            current_audio = AudioSegment.from_file(os.path.join(folder_path, file))
            combined += current_audio + silence

        combined.export(f'./{name}_full_3.wav', format="wav")

        break


# concatenate()
