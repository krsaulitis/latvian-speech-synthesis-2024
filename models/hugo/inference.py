import csv
import os
import time

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

base_url = os.getenv("HUGO_BASE_URL")
app_id = os.getenv("HUGO_APP_ID")
client_id = os.getenv("HUGO_CLIENT_ID")


def process_texts(text, out_path):
    post_url = f"{base_url}/ws/speech/synthesis"
    post_headers = {
        "Referer": f"{base_url}/lv/Speech/Synthesis",
        "Client-Id": client_id,
        "Host": base_url,
        "Origin": base_url,
        "X-Requested-With": "XMLHttpRequest"
    }

    post_data = {
        "appID": app_id,
        "text": text,
        "voice": "",
        "pitch": "1",
        "tempo": "1"
    }

    post_response = requests.post(post_url, headers=post_headers, data=post_data)

    if post_response.status_code == 200:
        request_id = post_response.json().get('request_id')
        if request_id:
            print(f"Request ID for '{text}': {request_id}")

            time.sleep(3)

            get_url = f"{base_url}/ws/speech/synthesis/{request_id}/audio"
            get_params = {
                "appID": app_id,
                "clientId": client_id
            }

            get_response = requests.get(get_url, params=get_params)

            if get_response.status_code == 200:
                with open(out_path, "wb") as audio_file:
                    audio_file.write(get_response.content)
                print(f"Audio file for '{text}' has been saved as '{out_path}'")
            else:
                print(f"Failed to retrieve the audio file for '{text}'. Status code: {get_response.status_code}")
        else:
            print(f"Request ID not found in the response for '{text}'.")
    else:
        print(f"POST request failed for '{text}'. Status code: {post_response.status_code}")


with open('./test_concatenated_w_phones.tsv', 'r') as file:
    rows = csv.reader(file, delimiter='\t')

    for i, row in tqdm(enumerate(rows, start=0), total=100):
        if i == 0:
            continue

        sentence = row[3]
        file = row[2]
        path = f'./audios_test/{file}'
        if os.path.exists(path):
            continue
        process_texts(sentence, path)



