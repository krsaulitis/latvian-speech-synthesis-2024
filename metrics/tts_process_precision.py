import glob
import os
import csv
import string
import argparse

import numpy as np
import pandas as pd

import miniaudio
import torch
import soundfile as sf
import librosa
import torchaudio
from loguru import logger
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    batch_size = args.batch_size
    base_path = '../asya-tts/precision_data'
    tts_model_id = args.model
    # df = pd.read_csv(f'./precision_data/meta.tsv', sep='\t')
    predictions_path = f'{base_path}/{tts_model_id}_predictions.tsv'

    model_id = f'./results/whisper-largev1-lv-run11-augm-punc/checkpoint-94000'

    logger.debug(f"loading whisper medium. Path: {model_id}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, local_files_only=True, low_cpu_mem_usage=True,
        use_safetensors=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, language="Latvian",
                                              task="transcribe")

    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=225,
        chunk_length_s=20,
        batch_size=batch_size,
        torch_dtype=torch.float32,
        device="cuda"
    )

    files = os.listdir(f'{base_path}/{tts_model_id}/')

    logger.debug(f'Found {len(files)} files in {base_path}/{tts_model_id}/')

    batch = []
    for file in files:
        if not file.endswith('.wav'):
            continue

        batch.append(f'{base_path}/{tts_model_id}/{file}')
        if len(batch) < batch_size:
            continue

        results = stt_pipe(batch, generate_kwargs={"language": "latvian", "task": "transcribe"})

        try:
            for i, result in enumerate(results):
                file_name = batch[i].split('/')[-1]
                sentence_idx = file_name.split('_')[0]
                step_idx = file_name.split('_')[1]
                out_text = result["text"].strip()

                logger.debug(f'Processed file: {file_name} with text: {out_text}')

                with open(predictions_path, 'a', encoding='utf-8', newline='\n') as f:
                    predictions_writer = csv.writer(f, delimiter='\t')
                    predictions_writer.writerow([sentence_idx, step_idx, out_text])

        except Exception as e:
            logger.error(f'Error processing batch: {batch}')
            logger.error(e)

        batch = []

        # sentence_idx = file.split('_')[0]
        # step_idx = file.split('_')[1]
        #
        # y, sr = librosa.load(f'{base_path}/{tts_model_id}/{file}', sr=None)
        # result_audio_part = stt_pipe(y, generate_kwargs={"language": "latvian", "task": "transcribe"})
        # out_text = result_audio_part["text"].strip()
        #
        # logger.debug(f'Processed file: {file} with text: {out_text}')
        #
        # with open(predictions_path, 'a', newline='\n') as f:
        #     predictions_writer = csv.writer(f, delimiter='\t')
        #     predictions_writer.writerow([sentence_idx, step_idx, out_text])

    # wer_metric = evaluate.load("wer")
    # cer_metric = evaluate.load("cer")

    # pred_str_l = ['...', '...']
    # label_str_l = ['...', '...']
    # wer = 100 * wer_metric.compute(predictions=pred_str_l, references=label_str_l)
    # cer = 100 * cer_metric.compute(predictions=pred_str_l, references=label_str_l)

    # y_part is numpy array wav
    # result_audio_part = stt_pipe(y_part, generate_kwargs={"language": "latvian", "task": "transcribe"})

    # out_text = result_audio_part["text"].strip()

    # df.to_csv(f'./precision_data/{tts_model_id}/meta.tsv', sep='\t', index=False)
