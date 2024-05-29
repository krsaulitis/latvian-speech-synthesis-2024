import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydub import AudioSegment, effects
from pydub.silence import detect_silence, detect_nonsilent, split_on_silence


# def deconcatenate():
#     data = pd.read_csv('./enhanced-missing/missing.tsv', sep='\t')
#     grouped = data.groupby('client_id')
#
#     for name, group in tqdm(grouped):
#         file_path = f'./enhanced-missing/enhanced/{name}-enhanced-90p.wav'
#         audio = AudioSegment.from_file(file_path)
#
#         current_time = 0
#         for index, row in group.iterrows():
#             def trim_silence(segment):
#                 margin = 200
#                 non_silence = detect_nonsilent(segment, min_silence_len=100, silence_thresh=segment.dBFS-16)
#                 trim_start = non_silence[0][0] - margin if non_silence[0][0] > margin else 0
#                 trim_end = non_silence[-1][1] + margin if non_silence[-1][1] + margin < len(segment) else len(segment)
#                 return segment[trim_start:trim_end]
#
#             full_path = row['path']
#             file = full_path.split('/')[-1]
#             file_without_extension = file.split('.')[0]
#
#             duration = row['audio_length']
#
#             segment = audio[current_time:current_time + duration]
#             current_time += duration + 300
#             segment = trim_silence(segment)
#
#             os.makedirs(f'./enhanced-missing/enhanced/{name}', exist_ok=True)
#             segment.export(f'./enhanced-missing/enhanced/{name}/{file_without_extension}.wav', format='wav')

def deconcatenate():
    data = pd.read_csv('./test_concatenated.tsv', sep='\t')
    file_path = f'./test_concatenated-enhanced-90p.wav'
    audio = AudioSegment.from_file(file_path)
    current_time = 0

    for index, row in tqdm(data.iterrows(), total=len(data)):
        full_path = row['path']
        file = full_path.split('/')[-1]

        duration = row['audio_length']

        segment = audio[current_time:current_time + duration]
        current_time += duration + 300

        os.makedirs(f'./test', exist_ok=True)
        segment.export(f'./test/{file}', format='wav')


deconcatenate()
exit()


def concatenate():
    data = pd.read_csv('./test_concatenated.tsv', sep='\t')
    # grouped = data.groupby('client_id')
    # client_count = len(grouped)

    silence = AudioSegment.silent(duration=300)
    combined = AudioSegment.silent(duration=0)

    for index, row in tqdm(data.iterrows(), total=len(data)):
        file_path = f'../files/test_non_enhanced/{row["path"]}'
        audio = AudioSegment.from_file(file_path)

        length = len(audio)
        data.at[index, 'audio_length'] = length

        combined += audio + silence

    combined.export(f'./test_concatenated.wav', format='wav')
    data.to_csv('./test_concatenated.tsv', sep='\t', index=False)

    # for name, group in grouped:
    #     client_count -= 1
    #     print(f"Processing {name} ({client_count} clients left)")
    #     folder_path = f'../../sets/common_voice/clips/{name}'
    #
    #     silence = AudioSegment.silent(duration=300)
    #     combined = AudioSegment.silent(duration=0)
    #
    #     for file_path in tqdm(group['path'], desc=f"Processing {name}"):
    #         file = file_path.split('/')[-1]
    #         current_audio = AudioSegment.from_file(os.path.join(folder_path, file))
    #         combined += current_audio + silence
    #
    #     combined.export(f'./{name}.wav', format="wav")


concatenate()
