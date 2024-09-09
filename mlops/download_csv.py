# script to download the file
import requests
import os
import config
import time
import yaml

# create the directory even if it exists
os.makedirs(config.EXTERNAL_DATA_DIR,exist_ok=True)

DOWNLOAD_PARAMS = config.get_parameters('data_download')

response = requests.get(DOWNLOAD_PARAMS.get('url'))
print(response)
if response.status_code == 200:
    with open(config.EXTERNAL_DATA_DIR / "tweet_emotions.csv", "wb") as file:
        file.write(response.content)
        print("CSV file downloaded successfully.")
else:
    response.raise_for_status()

    