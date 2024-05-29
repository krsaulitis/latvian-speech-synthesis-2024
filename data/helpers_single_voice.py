import os
import random
import shutil

import mutagen
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from pydub.silence import detect_nonsilent


ffmpeg_path = "/home/evaldsu/asya-tts/helpers/ffmpeg/ffmpeg-git-20240504-amd64-static"
if ffmpeg_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + ffmpeg_path

df = pd.read_csv('./sets/audio_books/data_lv_audio_books/metadata_ceplis_w_lengths.csv')
# df_lengths = pd.read_csv('./sets/common_voice/clip_durations.tsv', sep='\t')
folder_path = './sets/audio_books/data_lv_audio_books/ceplis_enhanced'
audio = None
audio_idx = 17
current_time = 0
total_duration = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    if idx < 4982:
        continue

    if not audio:
        audio = AudioSegment.from_file(os.path.join(folder_path, f'ceplis_{audio_idx}_full-enhanced-100p.mp3'))

    # duration = df_lengths[df_lengths['clip'] == row['path']]['duration[ms]'].values[0]
    duration = (row['length'] * 1000) - total_duration
    total_duration += duration

    def trim_silence(segment):
        margin = 200
        non_silence = detect_nonsilent(segment, min_silence_len=100, silence_thresh=segment.dBFS - 16)
        trim_start = non_silence[0][0] - margin if non_silence[0][0] > margin else 0
        trim_end = non_silence[-1][1] + margin if non_silence[-1][1] + margin < len(segment) else len(segment)
        return segment[trim_start:trim_end]

    segment = audio[current_time:current_time + duration]
    current_time += duration + 300
    segment = trim_silence(segment)

    segment.export(f'./sets/audio_books/split_ceplis_missing/{row["file_name"].replace(".flac", ".wav").replace(".mp3", ".wav")}', format="wav")

    if current_time >= (len(audio) - 300):
        audio_idx += 1
        current_time = 0
        total_duration = 0
        audio = None


exit()

folder_path = f'./sets/common_voice/clips'
df = pd.read_csv('./NISQA_results_deduplicated.csv')

silence = AudioSegment.silent(duration=300)
combined = AudioSegment.silent(duration=0)
current_len = 0
audio_idx = 1
max_len = 3400

for idx, row in tqdm(df.iterrows(), total=len(df)):
    if idx < 8672:
        continue

    client_id = row['client_id'][:6]
    audio_path = os.path.join(folder_path, client_id, row['path'])

    audio = mutagen.File(audio_path, easy=True)
    current_len += audio.info.length

    current_audio = AudioSegment.from_file(audio_path)
    combined += current_audio + silence

    # add length to df
    # df.at[idx, 'length'] = current_len

    if current_len > max_len:
        combined.export(f'./sets/common_voice_single/missing/cv_v17_{audio_idx}_full.mp3', format="mp3")
        audio_idx += 1
        current_len = 0
        combined = AudioSegment.silent(duration=0)


combined.export(f'./sets/common_voice_single/missing/cv_v17_{audio_idx}_full.mp3', format="mp3")
# df.to_csv('./sets/asya/data_lv_audio_books/metadata_ceplis_w_lengths.csv', index=False)
exit()

df = pd.read_csv('./NISQA_results.csv')
df = df.sort_values(by='mos_pred', ascending=False)
df = df.drop_duplicates(subset='sentence_id', keep='first')

df.to_csv('./NISQA_results_deduplicated.csv', index=False)

exit()


def common_voice_unique_text_files():
    base_path = './sets/common_voice/clips'

    df = pd.read_csv('./sets/common_voice/validated.tsv', sep='\t')
    df_deduplicated = df.drop_duplicates(subset='sentence_id', keep='first')

    enhanced_df = pd.read_csv('./enhancement/adobe/data_for_adobe_re_w_missing.tsv', sep='\t')
    enhanced_df = enhanced_df.drop_duplicates(subset='sentence_id', keep='first')

    non_enhanced_sentence_ids = []
    for idx, row in df_deduplicated.iterrows():
        if row['sentence_id'] not in enhanced_df['sentence_id'].values:
            non_enhanced_sentence_ids.append(row['sentence_id'])

    print(len(enhanced_df))
    print(len(non_enhanced_sentence_ids))
    print(len(df_deduplicated))

    non_enhanced_df = df[df['sentence_id'].isin(non_enhanced_sentence_ids)]
    print(len(non_enhanced_df))

    for idx, row in non_enhanced_df.iterrows():
        full_path = os.path.join(base_path, row['path'])
        non_enhanced_df.at[idx, 'path'] = full_path

    non_enhanced_df.to_csv('./non_enhanced_common_voice.csv', index=False)


# common_voice_unique_text_files()
# exit()


def sample_files(base_dir, num_samples=5):
    if not os.path.isdir(base_dir):
        raise ValueError(f"Provided path {base_dir} is not a valid directory.")

    for root, dirs, _ in os.walk(base_dir):
        for sub_dir in tqdm(dirs):
            sub_dir_path = os.path.join(root, sub_dir)

            all_files = []
            for sub_root, _, files in os.walk(sub_dir_path):
                for file in files:
                    if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                        all_files.append(os.path.join(sub_root, file))

            if len(all_files) <= num_samples:
                selected_files = all_files
            else:
                selected_files = random.sample(all_files, num_samples)

            print(f"Sampled files from {sub_dir_path}:")
            os.makedirs(f'./sampled_files/{sub_dir}', exist_ok=True)

            for file in selected_files:
                shutil.copy(file, f'./sampled_files/{sub_dir}')
                print(file)

        break


# sample_files('./enhancement')
sample_files('../beegfs/speech_data_16k')
