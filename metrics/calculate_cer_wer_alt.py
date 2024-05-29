import os
import csv
import re
from jiwer import wer, cer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def calculate_cer_wer(tsv_file):
    modified_rows = []

    df_meta = pd.read_csv('./meta/meta_raw.tsv', sep='\t')
    df_meta['id'] = df_meta['id'].astype(str)

    with open(tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        total_count = 0
        skip_count = 0
        cer_scores = []
        wer_scores = []

        checked_files = []

        for row in reader:
            try:
                total_count += 1

                prediction = normalize_text(row[2])
                file = row[3]

                truth = df_meta.loc[df_meta['path'] == file, 'text'].values[0]
                row[0] = row[3]
                row.pop(3)

                checked_files.append(file)

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

        unique_checked_files = set(checked_files)
        if len(unique_checked_files) != len(checked_files):
            print(f'Warning: Duplicate files in {tsv_file}')

        if total_count > 99:
            print(f'File: {tsv_file}')

        print(f'Skipped {skip_count}/{total_count}')
        # print(f'Average CER: {avg_cer}')
        # print(f'Average WER: {avg_wer}')

    return cer_scores, wer_scores, np.mean(cer_scores), np.mean(wer_scores)


path_list = [
    'predictions/raw-clarin_predictions.tsv',
    'predictions/raw-coqui_predictions.tsv',
    'predictions/raw-espeak_predictions.tsv',
    'predictions/raw-hugo_predictions.tsv',
    'predictions/raw-non-converted_predictions.tsv',
    'predictions/raw-non-enhanced_predictions.tsv',
    'predictions/test-adobe_predictions.tsv',
    'predictions/test-asya_predictions.tsv',
    'predictions/test-free-vc-female_predictions.tsv',
    'predictions/test-free-vc-male_predictions.tsv',
    'predictions/test-mp-senet_predictions.tsv',
    'predictions/test-resemble_predictions.tsv',
    'predictions/test-rvc-female_predictions.tsv',
    'predictions/test-rvc-male_predictions.tsv',
    'predictions/test-so-vits-female_predictions.tsv',
    'predictions/test-so-vits-male_predictions.tsv',
    'model-predictions/model-3/final',
    'model-predictions/model-6/final',
    'model-predictions/model-8/final/1_predictions.tsv',
]

model_metrics = {
    'cer': {},
    'wer': {},
}
model_speaker_metrics = {
    'cer': {},
    'wer': {},
}

for path in path_list:
    if path.endswith('.tsv'):
        _, _, avg_cer, avg_wer = calculate_cer_wer(f'{path}')
        model_metrics['cer'][path] = avg_cer
        model_metrics['wer'][path] = avg_wer
        model_speaker_metrics['cer'][path] = avg_cer
        model_speaker_metrics['wer'][path] = avg_wer

    if os.path.isdir(path):
        dir_cer = []
        dir_wer = []
        for speaker in os.listdir(path):
            _, _, avg_cer, avg_wer = calculate_cer_wer(f'{path}/{speaker}')
            dir_cer.append(avg_cer)
            dir_wer.append(avg_wer)
            model_speaker_metrics['cer'][f'{path}/{speaker}'] = avg_cer
            model_speaker_metrics['wer'][f'{path}/{speaker}'] = avg_wer
        model_metrics['cer'][path] = np.mean(dir_cer)
        model_metrics['wer'][path] = np.mean(dir_wer)


df_model = pd.DataFrame({
    'model': list(model_metrics['cer'].keys()),
    'cer': list(model_metrics['cer'].values()),
    'wer': list(model_metrics['wer'].values())
})
df_model.to_csv('cer_wer_all_results.csv', index=False)

df_speaker = pd.DataFrame({
    'model': list(model_speaker_metrics['cer'].keys()),
    'cer': list(model_speaker_metrics['cer'].values()),
    'wer': list(model_speaker_metrics['wer'].values())
})
df_speaker.to_csv('cer_wer_all_speaker_results.csv', index=False)

