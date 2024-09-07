# script to download the file
import requests
import os
import config
import time

os.makedirs(config.EXTERNAL_DATA_DIR,exist_ok=True)

url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
response = requests.get(url)
print(response)
if response.status_code == 200:
    with open(config.EXTERNAL_DATA_DIR / "tweet_emotions.csv", "wb") as file:
        file.write(response.content)
        print("CSV file downloaded successfully.")
else:
    response.raise_for_status()
    