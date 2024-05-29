import os
import csv
import re
from jiwer import wer, cer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_precision = pd.read_csv('./cer_wer_all_speaker_results.csv')
df_quality = pd.read_csv('./nisqa_model_all_speaker_results.tsv', sep='\t')

df_precision['speaker_id'] = df_precision['model'].apply(lambda x: x.split('/')[-1].split('_')[0] if x.startswith('model') else None)
df_precision['speaker_id'] = df_precision['speaker_id'].astype(float)
df_precision['model'] = df_precision['model'].apply(lambda x: x.split('_')[0]).apply(lambda x: x.split('/')[1])
df_precision = df_precision[['model', 'speaker_id', 'cer', 'wer']]

df_quality['model'] = df_quality['model'].apply(lambda x: x.split('/')[1] if x.startswith('models') else x)

# df_precision.to_csv('./cer_wer_all_speaker_results_1.csv', index=False)

if 'cer' not in df_quality.columns:
    df_quality['cer'] = float('nan')
if 'wer' not in df_quality.columns:
    df_quality['wer'] = float('nan')

df_quality['speaker_id'] = df_quality['speaker_id'].astype(float)
df_quality = df_quality[['model', 'speaker_id', 'cer', 'wer']]

df_concat = pd.concat([df_precision, df_quality],  ignore_index=True)
df_concat.to_csv('./all_speaker_results.csv', index=False)

exit()


def normalize_text(text):
    # remove non-alphabet characters
    normalized_text = ''.join([char if char.isalpha() else ' ' for char in text])
    # lowercase all letters
    normalized_text = normalized_text.lower()
    # remove multiple spaces
    while '  ' in normalized_text:
        normalized_text = normalized_text.replace('  ', ' ')

    # trim spaces
    normalized_text = normalized_text.strip()

    return normalized_text


def calculate_cer_wer(tsv_file, model_id):
    modified_rows = []

    if os.path.exists(f'./meta/meta_{model_id}.tsv'):
        df_meta = pd.read_csv(f'./meta/meta_{model_id}.tsv', sep='\t')
    else:
        df_meta = pd.read_csv('./meta/meta.tsv', sep='\t')

    df_meta['id'] = df_meta['id'].astype(str)

    with open(tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        total_count = 0
        skip_count = 0
        cer_scores = []
        wer_scores = []

        for row in reader:
            try:
                total_count += 1
                sentence_id = row[0]
                step_idx = row[1]
                if step_idx == '0':
                    continue

                prediction = normalize_text(row[2])
                file = row[3] if len(row) > 3 else None

                if file:
                    truth = df_meta.loc[df_meta['path'] == file, 'text'].values[0]
                    row[0] = row[3]
                    row.pop(3)
                else:
                    truth = df_meta.loc[df_meta['id'] == sentence_id, 'text'].values[0]

                if not truth:
                    skip_count += 1
                    continue

                truth = normalize_text(truth)

                if prediction:
                    row_cer = cer(prediction, truth)
                    row_wer = wer(prediction, truth)
                elif len(truth.split()) > 1:
                    row_cer = 1
                    row_wer = 1
                    print(f'Empty prediction: {prediction}, truth: {truth} for file: {row[1]}')
                else:
                    skip_count += 1
                    continue

                cer_scores.append(row_cer)
                wer_scores.append(row_wer)

                row.append(row_cer)
                row.append(row_wer)
                modified_rows.append(row)
            except Exception as e:
                skip_count += 1
                print('Error on row: ', row[0])
                print(e)

        avg_cer = np.mean(cer_scores)
        avg_wer = np.mean(wer_scores)

        print(f'Skipped {skip_count}/{total_count}')
        print(f'Average CER: {avg_cer}')
        print(f'Average WER: {avg_wer}')

    with open('./cer_wer.tsv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(modified_rows)


def process_model(model_id):
    print(f'Processing model: {model_id}')
    calculate_cer_wer(f'./predictions/{model_id}_predictions.tsv', model_id)

    os.makedirs(f'./{model_id}/by_sentence', exist_ok=True)

    df = pd.read_csv('./cer_wer.tsv', sep='\t', header=None)
    df.columns = ['sentence_id', 'step_idx', 'prediction', 'cer', 'wer']
    df = df.sort_values(by=['step_idx', 'sentence_id'], ascending=[True, True])
    df.to_csv(f'./{model_id}/cer_wer.tsv', sep='\t', index=False)

    if model_id in ['raw']:
        return

    data = pd.read_csv(f'./{model_id}/cer_wer.tsv', sep='\t')

    # Group data by sentence_id
    grouped_data = data.groupby('sentence_id')

    # Plot each sentence in its own graph
    for sentence_id, group in grouped_data:
        plt.figure(figsize=(12, 6))
        plt.plot(group['step_idx'], group['cer'], label=f'Sentence {sentence_id}', color='blue')
        plt.plot(group['step_idx'], group['wer'], label=f'Sentence {sentence_id}', color='red')
        plt.xlabel('Step')
        plt.ylabel('CER')
        plt.title(f'CER for Sentence {sentence_id} Over Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./{model_id}/by_sentence/cer_wer_{sentence_id}.png')
        plt.close()

    # Group by step_idx and get mean of cer and wer
    df_by_step = df.groupby('step_idx').agg({'cer': 'mean', 'wer': 'mean'}).reset_index()
    df_by_step.to_csv(f'./{model_id}/cer_wer_by_step.tsv', sep='\t', index=False)

    data = pd.read_csv(f'./{model_id}/cer_wer_by_step.tsv', sep='\t')

    print(f'Top 5 rows of {model_id} data:')
    print(data.sort_values(by='cer').head(5))
    print(data.sort_values(by='wer').head(5))


    plt.figure(figsize=(12, 6))
    plt.plot(data['step_idx'], data['cer'], label='CER', color='blue')
    plt.plot(data['step_idx'], data['wer'], label='WER', color='red')

    plt.title('CER un WER rādītāji katrā solī')
    plt.xlabel('Solis')
    plt.ylabel('Rādītāju vērtība')
    plt.legend()

    plt.grid(True)
    plt.savefig(f'./{model_id}/cer_wer_by_step.png')
    plt.close()


model_list = [
    'raw',
    'multi-v0.1',
    'multi-v0.2',
    'multi-v0.2-phon',
    'multi-v0.3',
    'multi-v0.4',
    'multi-v0.5',
    'single-v0.6',
    'single-v0.7',
    'single-v0.8',
]

for model_id in model_list:
    process_model(model_id)
