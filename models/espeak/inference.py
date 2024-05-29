import csv
import os

from tqdm.auto import tqdm


def generate_speech(text, output_file):
    command = f"espeak -v lv+male3 -s 120 '{text}' --stdout > {output_file}"
    os.system(command)


with open('./test_concatenated_w_phones.tsv', 'r') as file:
    rows = csv.reader(file, delimiter='\t')

    for i, row in tqdm(enumerate(rows, start=0), total=100):
        if i == 0:
            continue

        sentence = row[3]
        file = row[2]
        path = f'./audios_test/{file}'
        generate_speech(sentence, path)
