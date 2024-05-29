import csv
import shutil
import subprocess

import pandas as pd


def calculate_nisqa(input_dir, output_dir):
    subprocess.run(
        ['python', '../nisqa/run_predict.py',
         '--mode', 'predict_dir',
         '--pretrained_model', '../nisqa/weights/nisqa.tar',
         '--data_dir', input_dir,
         '--num_workers', '4',
         '--bs', '16',
         '--output_dir', output_dir]
    )
    shutil.move(f'{output_dir}/NISQA_results.csv', f'{output_dir}/NISQA_full_results.csv')

    subprocess.run(
        ['python', '../nisqa/run_predict.py',
         '--mode', 'predict_dir',
         '--pretrained_model', '../nisqa/weights/nisqa_tts.tar',
         '--data_dir', input_dir,
         '--num_workers', '4',
         '--bs', '16',
         '--output_dir', output_dir]
    )
    shutil.move(f'{output_dir}/NISQA_results.csv', f'{output_dir}/NISQA_tts_results.csv')

    data_full = pd.read_csv(f'{output_dir}/NISQA_full_results.csv')
    data_tts = pd.read_csv(f'{output_dir}/NISQA_tts_results.csv')

    results = [
        data_tts['mos_pred'].mean(),
        data_tts['mos_pred'].std(),
        data_tts['mos_pred'].min(),
        data_tts['mos_pred'].max(),
        data_tts['mos_pred'].median(),
    ]

    metrics = ['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']
    for metric in metrics:
        results.extend([
            data_full[metric].mean(),
            data_full[metric].std(),
            data_full[metric].min(),
            data_full[metric].max(),
            data_full[metric].median(),
        ])

    return results


base_dir = './precision_data/enhancement_and_conversion'
dirs = [
    'test-adobe',
    'test-asya',
    'test-free-vc-female',
    'test-free-vc-male',
    'test-mp-snet',
    'test-resemble',
    'test-rvc-female',
    'test-rvc-male',
    'test-so-vits-female',
    'test-so-vits-male',
]
cols = ['naturalness', 'quality', 'noise', 'distortion', 'coloration', 'loudness']
for col in cols:
    cols.extend([f'{col}_{metric}' for metric in ['mean', 'std', 'min', 'max', 'median']])

full_results = [cols]

for dir in dirs:
    full_results.append([dir] + calculate_nisqa(f'{base_dir}/{dir}', f'{base_dir}/{dir}'))

csv.writer(open('nisqa_results.tsv', 'w'), delimiter='\t').writerows(full_results)
