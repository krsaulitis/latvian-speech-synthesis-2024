import os

import numpy as np
import scipy
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from pyannote.audio import Pipeline
from tqdm import tqdm

load_dotenv()


if __name__ == "__main__":
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_AUTH_TOKEN"),

    # pipeline.to(torch.device("cuda"))

    # data_path = '../conversion/FreeVC/test_concatenated.tsv'
    # df = pd.read_csv(data_path, sep='\t')

    speakers = ['male', 'female']
    for speaker in speakers:
        base_dir_path = '../testing/test_non_converted'
        # converted_dir_path = f'../conversion/FreeVC/test-free-vc-{speaker}'
        #
        # distances = []
        # for file in tqdm(os.listdir(base_dir_path)):
        #     if not file.endswith('.wav'):
        #         continue
        #
        #     waveform, sr = torchaudio.load(f'{base_dir_path}/{file}')
        #     diarization = pipeline({"waveform": waveform, "sample_rate": sr}, return_embeddings=True, num_speakers=1)
        #     embedding = diarization[1]
        #
        #     c_waveform, c_sr = torchaudio.load(f'{converted_dir_path}/{file}')
        #     c_diarization = pipeline({"waveform": c_waveform, "sample_rate": c_sr}, return_embeddings=True, num_speakers=1)
        #     c_embedding = c_diarization[1]
        #
        #     distance = scipy.spatial.distance.cdist(np.array(embedding), np.array(c_embedding), metric='cosine')
        #
        #     df.loc[df['path'] == file, f'distance_to_original_for_{speaker}'] = distance[0][0]
        #     distances.append(distance[0][0])
        #
        # logger.info(f'Average distance: {np.mean(distances)}')

        if speaker == 'male':
            target_file = '../../data/enhancement/adobe/enhanced-full/32fabc-detailed/common_voice_lv_37757564.wav'
        else:
            target_file = '../../data/enhancement/adobe/enhanced-full/8b4d10-detailed/common_voice_lv_37769976.wav'

        t_waveform, t_sr = torchaudio.load(target_file)
        t_diarization = pipeline({"waveform": t_waveform, "sample_rate": t_sr}, return_embeddings=True, num_speakers=1)
        t_embedding = t_diarization[1]

        distances = []
        for file in tqdm(os.listdir(base_dir_path)):
            if not file.endswith('.wav'):
                continue

            waveform, sr = torchaudio.load(f'{base_dir_path}/{file}')
            diarization = pipeline({"waveform": waveform, "sample_rate": sr}, return_embeddings=True, num_speakers=1)
            embedding = diarization[1]

            distance = scipy.spatial.distance.cdist(
                np.array(embedding),
                np.array(t_embedding),
                metric='cosine'
            )

            # df.loc[df['path'] == file, f'distance_to_target_{speaker}'] = distance[0][0]
            distances.append(distance[0][0])

        logger.info(f'Average distance: {np.mean(distances)} for speaker: {speaker}')

    # df.to_csv(data_path, sep='\t', index=False)

    exit()
