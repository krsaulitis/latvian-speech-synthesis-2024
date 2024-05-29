import os
import librosa
import mutagen
import noisereduce as nr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from concurrent.futures import ProcessPoolExecutor, as_completed
from pydub import AudioSegment, effects
from pydub.silence import detect_silence, detect_nonsilent, split_on_silence
from tqdm import tqdm
import csv
import shutil

ffmpeg_path = "/opt/homebrew/bin"
if ffmpeg_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + ffmpeg_path


df1 = pd.read_csv('./file_dump/test.tsv', delimiter='\t')
df2 = pd.read_csv('./file_dump/test_single.tsv', delimiter='\t')
df3 = pd.read_csv('./file_dump/test_single_ceplis.tsv', delimiter='\t')

sample_df1 = df1.sample(n=33, random_state=42)
sample_df1['real_path'] = sample_df1['path']
sample_df1['path'] = sample_df1['path'].apply(lambda x: x.split('/')[-1])
sample_df2 = df2.sample(n=33, random_state=42)
sample_df3 = df3.sample(n=33, random_state=42)

concatenated_df = pd.concat([sample_df1, sample_df2, sample_df3], ignore_index=True)
concatenated_df = concatenated_df[['client_id', 'path', 'sentence']]
concatenated_df.to_csv('./testing/test_concatenated.tsv', sep='\t', index=False)

sample_df1['path'] = sample_df1['real_path']

out_dir = './testing/test_non_enhanced/'

# base_dir_1 = './enhancement/adobe/enhanced'
# base_dir_2 = './sets/common_voice_single/cv_17_split'
# base_dir_2 = './sets/common_voice_single/split'  # non converted files
# base_dir_3 = './sets/audio_books/split_ceplis'

base_dir_1 = './sets/common_voice/clips'
base_dir_2 = './sets/common_voice/clips'
base_dir_3 = './sets/audio_books/data_lv_audio_books/ceplis'

os.makedirs(out_dir, exist_ok=True)

# copy files from base_dir_1 based on sample_df1
for idx, row in tqdm(sample_df1.iterrows(), total=len(sample_df1)):
    file_path = os.path.join(base_dir_1, row['path'])
    file_path_alt = file_path.replace('.wav', '.mp3')
    if os.path.exists(file_path):
        shutil.copy(file_path, out_dir)
    elif os.path.exists(file_path_alt):
        audio, sr = librosa.load(file_path_alt, sr=None)
        file = file_path.split('/')[-1]
        sf.write(os.path.join(out_dir, file), audio, sr, format='wav')
    else:
        print(f"File {file_path} does not exist")


# copy files from base_dir_2 based on sample_df2
for idx, row in tqdm(sample_df2.iterrows(), total=len(sample_df2)):
    client_id = row['client_id']
    file_path = os.path.join(base_dir_2, row['path'])
    file_path_alt = f'{base_dir_2}/{client_id}/{row["path"]}'.replace('.wav', '.mp3')
    if os.path.exists(file_path):
        shutil.copy(file_path, out_dir)
    elif os.path.exists(file_path_alt):
        audio, sr = librosa.load(file_path_alt, sr=None)
        file = file_path.split('/')[-1]
        sf.write(os.path.join(out_dir, file), audio, sr, format='wav')
    else:
        print(f"File {file_path} does not exist")


# copy files from base_dir_3 based on sample_df3
for idx, row in tqdm(sample_df3.iterrows(), total=len(sample_df3)):
    file_path = os.path.join(base_dir_3, row['path'])
    fila_alt_path = file_path.replace('wav', 'flac')
    if os.path.exists(file_path):
        shutil.copy(file_path, out_dir)
    if os.path.exists(fila_alt_path):
        # convert to wav
        audio, sr = librosa.load(fila_alt_path, sr=None)
        file = file_path.split('/')[-1]
        sf.write(os.path.join(out_dir, file), audio, sr, format='wav')
    else:
        print(f"File {file_path} does not exist")

exit()



# def add_phonemes_to_dataset():
#     df = pd.read_csv('./testing/test_concatenated.tsv', sep='\t')
#     phoneme_file_path = './testing/transcriptions.transcribed.txt'
#     phoneme_map = {}
#     # with open('../comparison_models/vits-lv-clarin/config/phoneme_map.tsv', 'r', encoding='utf-8') as file:
#     #     for line in file:
#     #         parts = line.strip().split('\t')
#     #         if len(parts) == 2:
#     #             phoneme_map[parts[0]] = parts[1]
#
#
#     with open(phoneme_file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#         for idx, line in enumerate(lines):
#             text, text_phonemised = line.strip().split('\t')
#
#             text_phonemised_final = []
#
#             # words = text_phonemised.split(' ')
#             # for word in words:
#             #     phonemes = word.split('_')
#             #     for phoneme in phonemes:
#             #         phoneme_mapped = phoneme_map.get(phoneme, phoneme)
#             #         text_phonemised_final.append(phoneme_mapped)
#             #
#             #         # phoneme_mapped = phoneme_map[phoneme_map[0] == phoneme][1].values
#             #         # text_phonemised_final.append(phoneme_mapped)
#             #
#             #     text_phonemised_final.append(' ')
#             #
#             # df.at[idx, 'sentence_phon'] = ''.join(text_phonemised_final[:-1])
#             df.at[idx, 'sentence_phoned'] = text_phonemised
#             df.at[idx, 'id'] = idx + 1
#
#     df['id'] = df['id'].astype(int)
#     df = df[['id', 'client_id', 'path', 'sentence', 'sentence_phoned']]
#     df.to_csv('./testing/test_concatenated_w_phones.tsv', sep='\t', index=False)
#
#
# add_phonemes_to_dataset()
# exit()


# input_file = './testing/test_concatenated.tsv'
# output_file = './transcriptions.txt'
#
# with open(input_file, 'r', encoding='utf-8') as tsvfile:
#     tsvreader = csv.DictReader(tsvfile, delimiter='\t')
#
#     with open(output_file, 'w', encoding='utf-8') as txtfile:
#         for row in tsvreader:
#             sentence = row['sentence']
#             sentence = sentence.lower()
#
#             sentence = sentence.replace('–', '-')
#             sentence = sentence.replace('“', '"')
#             sentence = sentence.replace('”', '"')
#             sentence = sentence.replace('‘', "'")
#             sentence = sentence.replace('’', "'")
#             sentence = sentence.replace('—', '-')
#
#             sentence = sentence.replace('x', 'ks')
#
#             # remove punctuation
#             sentence = sentence.replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(';', '').replace('…', '')
#             sentence = sentence.replace('-', '')
#             sentence = sentence.replace('"', '').replace(':', '').replace("'", '')
#             txtfile.write(sentence + '\n')
#
# exit()


def validate_all_files_exist():
    df = pd.read_csv('./sets/audio_books/data_lv_audio_books/metadata_ceplis_w_lengths.csv')

    for idx, row in df.iterrows():
        path = row['file_name']
        if not os.path.exists(f'./sets/audio_books/split_ceplis/{path.replace(".flac", ".wav").replace(".mp3", ".wav")}'):
            print(f"File {path} does not exist")


# validate_all_files_exist()
# exit()


def copy_fixes():
    df_fixed = pd.read_csv('./NISQA_results_deduplicated.csv')
    df = pd.read_csv('./sets/common_voice_single/cv_17_single_full.tsv', delimiter='\t')

    for idx, row in df.iterrows():
        fixed_sentence = df_fixed[df_fixed['path'] == row['path']]['sentence'].values

        df.at[idx, 'sentence'] = fixed_sentence[0]

    df.to_csv('./sets/common_voice_single/cv_17_single_full_fixed.tsv', sep='\t', index=False)


# copy_fixes()
# exit()


def fix_books_dataset():
    df = pd.read_csv('./sets/asya/data_lv_audio_books/metadata.csv')

    new_data = [['file_name', 'transcription', 'file_len', 'text_cer']]

    idx_adjustment = 0
    prev_text = None
    current_chapter = None
    current_fragment = None

    for idx, row in df.iterrows():
        text = row['transcription']
        file_name = row['file_name']

        title = "".join(file_name.split('_')[:-2])
        chapter = int(file_name.split('_')[-2])
        fragment = int(file_name.split('_')[-1].replace('.flac', ''))

        if current_chapter != chapter:
            idx_adjustment = 0
            current_fragment = fragment - 1

        if text == prev_text:
            idx_adjustment += 1
            print(f"Adjusting index for {file_name}")
            continue

        prev_text = text

        current_chapter = chapter
        current_fragment = current_fragment + 1

        file_name_start = file_name.split('_')[:-2]
        file_name = f'{title}_{current_chapter}_{current_fragment}.flac'

        len = row['file_len']
        cer = row['text_cer']

        new_data.append([file_name, text, len, cer])

    new_df = pd.DataFrame(new_data[1:], columns=new_data[0])
    new_df.to_csv('./sets/asya/data_lv_audio_books/metadata_fixed.csv', index=False)


# fix_books_dataset()
# exit()



def add_phonemes_to_train_test_split():
    df = pd.read_csv('./enhancement/adobe/data_for_adobe_re_w_missing_phon.tsv', sep='\t')
    train_df = pd.read_csv('./enhancement/adobe/test.tsv', sep='\t')

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        norm_path = row['path'].replace('.mp3', '.wav')
        if norm_path in train_df['path'].values:
            train_df.at[train_df[train_df['path'] == norm_path].index[0], 'sentence_phon'] = row['sentence_phon']

    train_df.to_csv('./enhancement/adobe/test_phon.tsv', sep='\t', index=False)


# add_phonemes_to_train_test_split()
# exit()



def get_random_files(num_files=1000):
    data = pd.read_csv('./enhancement/adobe/data_for_adobe.tsv', delimiter='\t')
    random_files = data.sample(num_files)

    for index, row in random_files.iterrows():
        print(row['path'])

    random_files.to_csv('./enhancement/adobe/enhancement_test_files.tsv', sep='\t', index=False)


# get_random_files()
# exit()


def get_folder_audio_length(directory_path='./sets/common_voice/clips'):
    is_recursive = True
    dir_audio_length = 0
    lengths = {}

    for file in tqdm(os.listdir(directory_path), desc=f"Processing {directory_path}", leave=False):
        if is_recursive and os.path.isdir(os.path.join(directory_path, file)):

            sub_dir_audio_length, sub_dir_lengths = get_folder_audio_length(os.path.join(directory_path, file))
            lengths[os.path.join(directory_path, file)] = sub_dir_audio_length

            lengths.update(sub_dir_lengths)

            # total_length += get_folder_audio_length(os.path.join(directory_path, file))
            continue

        if file.endswith('.mp3') or file.endswith('.wav') or file.endswith('.flac'):
            audio_path = os.path.join(directory_path, file)
            try:
                audio = mutagen.File(audio_path, easy=True)
                dir_audio_length += audio.info.length
            except Exception as e:
                print(f"Error processing file {audio_path}: {e}")

    print(f"Total audio length in {directory_path}: {dir_audio_length / 60} minutes or {dir_audio_length / 60 / 60} hours")
    print(f"Total audio length: {dir_audio_length / 60} minutes or {dir_audio_length / 60 / 60} hours")

    return dir_audio_length, lengths


# length, lengths = get_folder_audio_length('./sets/common_voice/clips')
#
# # sort
# lengths_sorted = sorted(lengths.items(), key=lambda x: x[1], reverse=True)
# for key, value in lengths_sorted[:20]:
#     print(f"{key}: {value / 60} minutes or {value / 60 / 60} hours")

# get_folder_audio_length('./single_speaker/split_ceplis')
# exit()


def get_top_directories_by_length():
    duration, lengths = get_folder_audio_length()
    sorted_lengths = {k: v for k, v in sorted(lengths.items(), key=lambda item: item[1], reverse=True)}
    for key, value in list(sorted_lengths.items())[:10]:
        print(f"{key}: {value / 60} minutes")


# get_top_directories_by_length()
# exit()


def resample():
    import os
    import librosa
    import soundfile as sf

    in_sr = 44100
    out_sr = 22050

    input_path = './test_files_for_speech_enhancement'
    output_path = './test_files_for_speech_enhancement/22khz'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_path in os.listdir(input_path):
        if "resemble" in file_path and file_path.endswith(".wav"):
            input_file = os.path.join(input_path, file_path)
            output_file = os.path.join(output_path, file_path)

            audio, sr = librosa.load(input_file, sr=in_sr)
            resampled_audio = librosa.resample(audio, orig_sr=in_sr, target_sr=out_sr)
            sf.write(output_file, resampled_audio, out_sr)


# resample()
# exit()


def train_test_split():
    df = pd.read_csv('./sets/audio_books/data_lv_audio_books/metadata_ceplis_w_lengths.csv')

    # Split the data into train and test sets
    train = df.sample(frac=0.99, random_state=42)
    test = df.drop(train.index)

    train['client_id'] = 1
    test['client_id'] = 1

    train.to_csv('./sets/common_voice_single/train_single_ceplis.tsv', sep='\t', index=False)
    test.to_csv('./sets/common_voice_single/test_single_ceplis.tsv', sep='\t', index=False)


train_test_split()
exit()


def build_dataset():
    data = pd.read_csv('./common_voice/updated_validated_w_real_length_nisqa.tsv', delimiter='\t')

    data['audio_length_min'] = data['audio_length_real'] / 60000
    data['client_id'] = data['client_id'].apply(lambda x: x[:6])

    # Average of mos_pred	noi_pred	dis_pred	col_pred	loud_pred
    data['quality'] = data[['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']].mean(axis=1)

    data_sorted = data.sort_values(by='mos_pred', ascending=False)

    grouped = data_sorted.groupby('client_id')

    filtered_clients = pd.DataFrame()
    # Iterate over each group
    for name, group in grouped:
        group['cumulative_audio_length'] = group['audio_length_min'].cumsum()
        valid_records = group[group['cumulative_audio_length'] <= 30]

        if valid_records['audio_length_min'].sum() >= 15:
            filtered_clients = pd.concat([filtered_clients, valid_records])

    filtered_clients['gender'] = filtered_clients['gender'].replace(
        {'male_masculine': 'male', 'female_feminine': 'female'})

    client_id_to_gender = {
        'd5bcc7': 'female',
        '2266fd': 'female',
        '30456c': 'female',
        'c5c18d': 'male',
        '06da81': 'female',
        'be3f03': 'female',
        '2a7d89': 'male',
        '332b0e': 'female',
        'eb5f38': 'female',
        'd63d1f': 'female',
        'd6944d': 'male',
        '94b2ac': 'female',
        '7b68c5': 'male',
        'f49187': 'female',
        '7b2e48': 'female',
        '30b015': 'female',
        '5658bb': 'female',
        'f7c950': 'female',
        'c39924': 'female',
        '28ff9b': 'female',
        'ffb8ae': 'female',
        '169982': 'female',
        'b4090f': 'male',
        '77fab6': 'female',
        '7806c2': 'female',
        '62d8ad': 'female',
        '7777f7': 'female',
        '126d55': 'male',
        '01afc6': 'female',
        '8f7663': 'female',
        '8443be': 'female',
        '7d160c': 'female',
        'cc89c4': 'female',
        '99e4c4': 'female',
    }

    filtered_clients.loc[filtered_clients['gender'].isnull(), 'gender'] = filtered_clients['client_id'].map(
        client_id_to_gender)

    # Calculate total audio length and average mos_pred for each client
    summary = filtered_clients.groupby('client_id').agg(
        length=('audio_length_min', 'sum'),
        avg_quality=('quality', 'mean'),
        avg_mos=('mos_pred', 'mean'),
        avg_noi=('noi_pred', 'mean'),
        avg_dis=('dis_pred', 'mean'),
        avg_col=('col_pred', 'mean'),
        gender=('gender', 'first'),
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # Print results for each client
        summary_by_length = summary.sort_values(by='length', ascending=False)
        print(summary_by_length)

        summary_by_mos = summary.sort_values(by='avg_mos', ascending=False)
        print(summary_by_mos)

        # Calculate and print the total audio length of the filtered section
        total_audio_length = filtered_clients['audio_length_min'].sum()
        total_mos_pred = filtered_clients['mos_pred'].mean()
        print("Total audio length of the filtered section:", total_audio_length / 60, "hours")
        print("Average MOS prediction of the filtered section:", total_mos_pred)

        # Female client count by gender column
        female_clients = filtered_clients[filtered_clients['gender'] == 'female']['client_id'].unique()
        male_clients = filtered_clients[filtered_clients['gender'] == 'male']['client_id'].unique()

        print(f"Females: {len(female_clients)}, Males: {len(male_clients)}")

        """
        When sorted by length:
        Total audio length of the filtered section: 57.80742166666666 hours
        Average MOS prediction of the filtered section: 3.4041115665535258

        When sorted by MOS:
        Total audio length of the filtered section: 57.810471388888885 hours
        Average MOS prediction of the filtered section: 3.6332488130555114
        """

    # Save the filtered dataset to a new TSV file
    filtered_clients = filtered_clients.drop(columns=['cumulative_audio_length', 'audio_length_min', 'audio_length'])
    filtered_clients['audio_length'] = filtered_clients['audio_length_real']
    # filtered_clients.to_csv('./common_voice/dataset_real.tsv', sep='\t', index=False)


build_dataset()
exit()


def append_audio_length_column(tsv_file_path, output_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')

    def get_audio_length(audio_path):
        audio_path = './common_voice/enhanced_clips/' + audio_path.replace('.mp3', '.wav')
        try:
            audio = mutagen.File(audio_path, easy=True)
            return int(float(audio.info.length) * 1000)  # Convert seconds to milliseconds
        except Exception as e:
            # print(f"Error processing file {audio_path}: {e}")
            return 0

    df['audio_length_real'] = df['path'].apply(get_audio_length)
    df.to_csv(output_file_path, sep='\t', index=False)


append_audio_length_column(
    './common_voice/updated_validated_w_length_nisqa.tsv',
    './common_voice/updated_validated_w_real_length_nisqa.tsv',
)
exit()


def plot_audio(audio):
    # Plot the audio dBFS over time
    chunk_length_ms = 20
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    dBFS_values = [chunk.dBFS for chunk in chunks]
    # Time axis for plotting (in seconds)
    time_axis = np.arange(0, len(audio) / 1000, chunk_length_ms / 1000)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, dBFS_values)
    plt.title('Audio dBFS Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('dBFS')
    plt.grid(True)
    plt.show()


def enhance_audio(audio_path, output_path, show_plots=False):
    audio_raw, sr = librosa.load(audio_path, sr=None)
    # Reduce noise
    audio = nr.reduce_noise(y=audio_raw, sr=sr)

    # meter = pyln.Meter(sr)
    # loudness = meter.integrated_loudness(audio)
    # audio = pyln.normalize.loudness(audio, loudness, -23.0)

    audio_int = np.int16(audio * 32767)
    audio = AudioSegment(
        audio_int.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16 bits / 8 = 2 bytes per sample
        channels=1
    )

    # Normalize audio
    audio = effects.compress_dynamic_range(audio)
    audio = effects.normalize(audio)

    # Check clipping
    if audio.max_dBFS > 0:
        print(f"Clipping detected in {audio_path}")

    # Trim silence and mouse clicks
    dBFS = audio.dBFS
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=200, silence_thresh=dBFS - 48)

    max_click_duration = 400
    first_click = nonsilent_ranges[0]
    if first_click[1] - first_click[0] < max_click_duration and len(nonsilent_ranges) > 1:
        nonsilent_ranges = nonsilent_ranges[1:]
    else:
        first_click = None

    last_click = nonsilent_ranges[-1]
    if last_click[1] - last_click[0] < max_click_duration and len(nonsilent_ranges) > 1:
        nonsilent_ranges = nonsilent_ranges[:-1]
    else:
        last_click = None

    margin_start = 400
    margin_end = 800
    if nonsilent_ranges:
        start_index = nonsilent_ranges[0][0] - margin_start if nonsilent_ranges[0][0] > margin_start else 0
        end_index = nonsilent_ranges[-1][1] + margin_end if nonsilent_ranges[-1][1] + margin_end < len(audio) else len(
            audio)
        if last_click:
            last_click_start = last_click[1] - 100
            end_index = end_index if end_index < last_click_start else last_click_start
    else:
        start_index, end_index = 0, len(audio)

    warn_threshold = 1300
    if start_index > warn_threshold:
        print(f"Removed {start_index / 1000} seconds of silence at the beginning of {audio_path}")

    if len(audio) - end_index > warn_threshold:
        print(f"Removed {(len(audio) - end_index) / 1000} seconds of silence at the end of {audio_path}")

    if show_plots:
        plot_audio(audio)
    audio = audio[start_index:end_index]
    if show_plots:
        plot_audio(audio)

    audio.export(output_path, format="wav")
    # return audio, output_path


def process_file(f):
    try:
        return enhance_audio(f, f.replace('clips', 'enhanced_clips').replace('.mp3', '.wav'))
    except Exception as e:
        print(f"Error processing file {f}: {e}")
        return None


# enhance_audio('./common_voice/clips/bbcac5/common_voice_lv_38256143.mp3', './common_voice/enhanced_clips/bbcac5/common_voice_lv_38256143.wav', True);
# enhance_audio('./common_voice/clips/ddaa71/common_voice_lv_39368937.mp3', './common_voice/enhanced_clips/ddaa71/common_voice_lv_39368937.wav', True);
# exit()

# def main():
#     files = []
#     for d_client in os.listdir('./common_voice/clips'):
#         dir_path = os.path.join('./common_voice/clips', d_client)
#         if not os.path.isdir(dir_path):
#             continue
#
#         if d_client == '_other':
#             continue
#
#         for f in os.listdir(dir_path):
#             if f.endswith('.mp3'):
#                 new_dir = dir_path.replace('clips', 'enhanced_clips')
#                 if not os.path.exists(new_dir):
#                     os.makedirs(os.path.dirname(new_dir), exist_ok=True)
#
#                 files.append(os.path.join(dir_path, f))
#
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = [executor.submit(process_file, f) for f in files]
#
#         for future in as_completed(futures):
#             audio, filename = future.result()
#             if audio:
#                 audio.export(filename, format="wav")
#
#             # print(future.result())
#
#
# if __name__ == '__main__':
#     main()
#     exit()

dataset = pd.read_csv('./common_voice/dataset.tsv', delimiter='\t')
client_ids = dataset['client_id'].unique()

files = []
for d_client in os.listdir('./common_voice/clips'):
    dir_path = os.path.join('./common_voice/clips', d_client)
    if not os.path.isdir(dir_path):
        continue

    if d_client not in client_ids:
        continue

    for f in os.listdir(dir_path):
        if f.endswith('.mp3'):
            files.append(os.path.join(dir_path, f))

i = 0
for f in tqdm(files):
    new_dir = f.replace('clips', 'enhanced_clips').replace(f.split('/')[-1], '')
    if not os.path.exists(new_dir):
        os.makedirs(os.path.dirname(new_dir), exist_ok=True)

    enhance_audio(f, f.replace('clips', 'enhanced_clips').replace('.mp3', '.wav'))

exit()


def append_nisqa_scores():
    nisqa_scores = pd.read_csv('./common_voice/NISQA_results.csv')
    nisqa_scores = nisqa_scores.drop(columns=['deg', 'model'])

    df = pd.read_csv('./common_voice/updated_validated_w_length.tsv', sep='\t')

    df = df.merge(nisqa_scores, on='path', how='left')
    df.to_csv('./common_voice/updated_validated_w_length_nisqa.tsv', sep='\t', index=False)


# append_nisqa_scores()
# exit()


def extract_possible_split():
    df = pd.read_csv('./common_voice/updated_validated_w_length.tsv', sep='\t')


def extract_client_auto_lengths():
    with open('./common_voice/client_durations.txt', 'r') as file:
        lines = file.readlines()

    def extract_data(line):
        parts = line.strip().split('/')
        voice_id = parts[-1].split(': ')[0]
        duration = float(parts[-1].split(': ')[1].split(' ')[0])
        duration = 35 if duration > 35 else duration
        return voice_id, duration

    all_data = [extract_data(line) for line in lines]

    # Filter out entries with less than 15 minutes of recordings
    filtered_data = [data for data in all_data if data[1] >= 20]
    sorted_data = sorted(filtered_data, key=lambda x: x[1], reverse=True)

    for client_id, audio_length in sorted_data:
        print(f"{client_id}: {audio_length} minutes")

    print(f"Total clients: {len(sorted_data)}")
    print(f"Total audio length: {(sum([x[1] for x in sorted_data])) / 60} hours")


# extract_client_auto_lengths()
# exit()


def unique_clients(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')

    df['client_id'] = df['client_id'].apply(lambda x: x[:6])

    print(f"Unique clients: {df['client_id'].nunique()}")


def audio_recordings_per_client(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')

    df['client_id'] = df['client_id'].apply(lambda x: x[:6])

    audio_by_client = df.groupby('client_id')['path'].count().reset_index()
    top_clients = audio_by_client.sort_values(by='path', ascending=False).head(300)

    print(f"Total audio recordings: {audio_by_client['path'].sum()}")
    print(f"Average audio recordings per client: {audio_by_client['path'].mean()}")
    print(f"Median audio recordings per client: {audio_by_client['path'].median()}")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(top_clients.head(100))


def total_length_of_audio(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')
    df['audio_length'] = df['audio_length'] / 1000 / 60 / 60

    # audio_by_client = df.groupby('client_id')['audio_length'].sum().reset_index()
    # top_clients = audio_by_client.sort_values(by='audio_length', ascending=False).head(200)

    df = df.sort_values(by=['client_id', 'audio_length'], ascending=[True, False])
    audio_by_client = df.groupby('client_id').head(1500)
    audio_by_client = audio_by_client.groupby('client_id')['audio_length'].sum().reset_index().head(100)
    # top_clients = audio_by_client['audio_length'].sum()

    print(f"Total length of audio: {audio_by_client['audio_length'].sum()} hours")


def median_client_audio_length(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')
    df['audio_length'] = df['audio_length'] / 1000 / 60
    print(f"Median client audio length: {df.groupby('client_id')['audio_length'].sum().median()} minutes")
    print(f"Mean client audio length: {df.groupby('client_id')['audio_length'].sum().mean()} minutes")


def plot_top_clients_audio(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')

    df['audio_length'] = df['audio_length'] / 1000 / 60

    audio_by_client = df.groupby('client_id')['audio_length'].sum().reset_index()

    audio_by_client['client_id'] = audio_by_client['client_id'].apply(lambda x: x[:6])

    top_clients = audio_by_client.sort_values(by='audio_length', ascending=False).head(200)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(top_clients.head(100))

    # Plotting.
    # plt.figure(figsize=(10, 32))
    # plt.barh(top_clients['client_id'], top_clients['audio_length'])
    # plt.xlabel('Total Length of Audio (minutes)')
    # plt.ylabel('Client ID')
    # plt.title('Top 100 Clients by Recorded Audio Length')
    # plt.tight_layout()
    # plt.show()


# audio_recordings_per_client('./common_voice/updated_validated_w_length.tsv')
# median_client_audio_length('./common_voice/validated_w_length.tsv')
# unique_clients('./common_voice/validated_w_length.tsv')
# total_length_of_audio('./common_voice/validated_w_length.tsv')
# plot_top_clients_audio('./common_voice/validated_w_length.tsv')
exit()


def regroup_files():
    file_name = './common_voice/validated_w_length.tsv'
    root_dir = './common_voice/clips'

    df = pd.read_csv(file_name, sep='\t')

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        subfolder_name = row['client_id'][:6]
        subfolder_path = os.path.join(root_dir, subfolder_name)

        # Check if the subfolder exists, if not create it
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Build the old and new paths
        old_path = os.path.join(root_dir, row['path'])
        new_path = os.path.join(subfolder_path, os.path.basename(row['path']))

        # Move the file to the new location
        # shutil.move(old_path, new_path)

        # Update the 'path' in the DataFrame to the new location
        df.at[index, 'path'] = os.path.relpath(new_path, root_dir)

    # Write the updated DataFrame to a new TSV file
    updated_file_name = './common_voice/updated_validated_w_length.tsv'
    df.to_csv(updated_file_name, sep='\t', index=False)

    print(f"Updated TSV file saved as {updated_file_name}.")

