import os

import shutil
import pandas as pd
from pydub import AudioSegment

# new_data = pd.read_csv('./enhanced-missing/missing_re.tsv', sep='\t')
# current_data = pd.read_csv('./data_for_adobe_re.tsv', sep='\t')
#
# grouped = current_data.groupby('client_id')
# for name, group in grouped:
#     current_length = group['audio_length'].sum()
#
#     new_group = new_data[new_data['client_id'] == name]
#     if len(new_group) == 0:
#         continue
#
#     remaining_length = new_group['audio_length'].sum()
#
#     print(f"{name}: {current_length / 60} minutes, {remaining_length / 60} minutes")
#
#     for idx, row in new_group.iterrows():
#         current_length += row['audio_length']
#         remaining_length -= row['audio_length']
#
#         if current_length > 1860:
#             break
#         current_data = current_data._append(row)
#
#         if os.path.exists(f"./enhanced/{row['path'].replace('.mp3', '.wav')}"):
#             print(f"File already exists: {row['path']}")
#             continue
#
#         # shutil.copy(
#         #     f"./enhanced-missing/enhanced/{row['path'].replace('.mp3', '.wav')}",
#         #     f"./enhanced/{row['path'].replace('.mp3', '.wav')}",
#         # )
#
#     print(f"{name}: {current_length / 60} minutes, {remaining_length / 60} minutes")
#     print("-----------------------------------------------------")
#
# current_data = current_data.sort_values(by=['client_id', 'mos_pred'], ascending=[True, False])
# current_data.to_csv('./data_for_adobe_re_w_missing.tsv', sep='\t', index=False)
#
# exit()

# for dir in os.listdir('./enhanced'):
#     if not os.path.isdir(f'./enhanced/{dir}'):
#         continue
#
#     enhanced_files = os.listdir(f'./enhanced/{dir}')
#     og_files = os.listdir(f'../../sets/common_voice/clips/{dir}')
#     og_files = [file.replace('.mp3', '.wav') for file in og_files]
#
#     # filter pt files
#     enhanced_files = [file for file in enhanced_files if file.endswith('.wav')]
#     og_files = [file for file in og_files if file.endswith('.wav')]
#
#     diff = set(og_files) - set(enhanced_files)
#
#     if len(diff) > 0:
#         print(f"{dir} has {len(diff)} additional files")

og_data = pd.read_csv('../../sets/common_voice/updated_validated_w_length_nisqa.tsv', delimiter='\t')
og_data['original_path'] = og_data['path']
og_data['path'] = og_data['path'].apply(lambda x: x.replace('.mp3', '.wav').split('/')[-1])
data = pd.read_csv('./data_for_adobe_re.tsv', delimiter='\t')
grouped = data.groupby('client_id')

missing_data = []

for name, group in grouped:
    total_length = group['audio_length'].sum()
    print(f"{name}: {total_length / 60} minutes")

    length_margin = 300
    required_length = 1800 + length_margin - total_length

    enhanced_files = os.listdir(f'./enhanced/{name}')
    og_files = os.listdir(f'../../sets/common_voice/clips/{name}')
    og_files = [file.replace('.mp3', '.wav') for file in og_files]
    enhanced_files = [file for file in enhanced_files if file.endswith('.wav')]
    og_files = [file for file in og_files if file.endswith('.wav')]

    diff = set(og_files) - set(enhanced_files)

    if len(diff) > 0:
        print(f"{name} has {len(diff)} additional files")

        diff_og_data = og_data[og_data['path'].isin(diff)]
        diff_og_data = diff_og_data.sort_values(by='mos_pred', ascending=False)

        silence = AudioSegment.silent(duration=300)
        combined = AudioSegment.silent(duration=0)

        for idx, row in diff_og_data.iterrows():
            if required_length <= 0:
                break

            # audio = AudioSegment.from_file(f'../../sets/common_voice/clips/{row["original_path"]}')
            # combined += audio + silence
            required_length -= row['audio_length'] / 1000
            missing_data.append(row)

        # combined.export(f'./enhanced-missing/{name}.wav', format='wav')

missing_data = pd.DataFrame(missing_data)

missing_data['client_id'] = missing_data['client_id'].apply(lambda x: x[:6])
missing_data['path'] = missing_data['original_path']
missing_data['quality'] = missing_data[['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']].mean(axis=1)
missing_data.drop(columns=['original_path'], inplace=True)

missing_data.to_csv(f'./enhanced-missing/missing_a.tsv', sep='\t', index=False)

exit()

# data['audio_length_minutes'] = data['audio_length'] / 60000
# data['client_id'] = data['client_id'].apply(lambda x: x[:6])
# data_sorted = data.sort_values(by='mos_pred', ascending=False)
#
# grouped = data.groupby('client_id')
# for name, group in grouped:
#     if name not in data_real['client_id'].values:
#         continue
#
#     total_length = 0
#     files_to_add = []
#     for idx, row in group.iterrows():
#         length = 0
#         if row['path'] not in data_real['path'].values:
#             length = row['audio_length'] / 1000 / 60
#         else:
#             length = data_real[data_real['path'] == row['path']]['audio_length'].values[0]
#
#         total_length += length
#
#         if total_length > 2400:
#             break
#
#         full_path = os.path.join('./enhanced', row['path'].replace('.mp3', '.wav'))
#         if os.path.exists(full_path):
#             data.drop(idx, inplace=True)
#             continue
#
#         files_to_add.append(row['path'])
#
#     silence = AudioSegment.silent(duration=300)
#     combined = AudioSegment.silent(duration=0)
#
#     for file in files_to_add:
#         audio = AudioSegment.from_file(f'../../sets/common_voice/clips/{file.replace(".wav", ".mp3")}')
#         combined += audio + silence
#
#     combined.export(f'./enhanced-missing/{name}.wav', format='wav')
