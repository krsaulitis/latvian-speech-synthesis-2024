import csv
from tqdm.auto import tqdm
from TTS.api import TTS

vits = TTS("tts_models/lv/cv/vits")


def generate():
    with open('./test_concatenated_w_phones.tsv', 'r') as file:
        rows = csv.reader(file, delimiter='\t')

        for i, row in tqdm(enumerate(rows, start=0), total=100):
            if i == 0:
                continue

            sentence = row[3]
            file = row[2]
            path = f'./audios_test/{file}'
            vits.tts_to_file(sentence, file_path=path)


generate()
