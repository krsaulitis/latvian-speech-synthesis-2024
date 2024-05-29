import os
import time

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def task_submit(file_path):
    with open(file_path, 'rb') as file:
        task_response = requests.post(
            url=os.getenv("ASYA_API_BASE_URL") + "/task_submit",
            params={
                "api_key": os.getenv("ASYA_API_KEY"),
                "features": ["audio_denoise"],
                "known_and_unknown_users_count": 1,
                "is_save_source_file_after_processing": True,
                "language_codes": ["lv"],
            },
            files={'file': file},
        )

        return task_response.json()


def task_status(task_id):
    task_response = requests.post(
        url=os.getenv("ASYA_API_BASE_URL") + "/task_status",
        params={
            "api_key": os.getenv("ASYA_API_KEY"),
            "task_uuid": task_id,
        },
    )

    return task_response.json()


def task_download(task_id):
    task_response = requests.get(
        url=os.getenv("ASYA_API_BASE_URL") + "/task_audio_denoised",
        params={
            "api_key": os.getenv("ASYA_API_KEY"),
            "task_uuid": task_id,
        },
    )

    return task_response.content


def denoise_audio(in_path, out_path, depth=0):
    if depth > 1:
        print(f"File {in_path} failed to denoise")
        return

    response = task_submit(in_path)
    print(response)
    is_success = response['is_success']
    task_id = response["task_uuid"]
    if not is_success:
        print(f"Task submission failed for file {in_path}")
        exit(1)
    print(f"Task submitted for file {in_path} with task_id {task_id}")

    request_count = 0
    is_ready = False
    while not is_ready:
        request_count += 1
        if request_count > 10:
            print(f"Task {task_id} is taking too long to complete")
            break
        response = task_status(task_id)
        print(response)
        status = response['request_status']

        if status == 'ERROR':
            print(f"Task {task_id} failed")
            exit(1)

        if status == 'READY':
            is_ready = True
            print(f"Task {task_id} is completed")
        else:
            time.sleep(request_count * 3)

    if not is_ready:
        denoise_audio(in_path, out_path, depth + 1)
        return

    denoised_file = task_download(task_id)
    with open(out_path, 'wb') as file:
        file.write(denoised_file)


def main():
    base_dir = '../files/test_non_enhanced'
    data = pd.read_csv('./test_concatenated.tsv', sep='\t')

    for index, row in data.iterrows():
        file_path = row['path']

        if os.path.exists(f"./test-asya/{file_path}"):
            print(f"File {file_path} already exists")
            continue

        denoise_audio(f"{base_dir}/{file_path}", f"./test-asya/{file_path}")
        # time.sleep(5)


if __name__ == "__main__":
    main()
