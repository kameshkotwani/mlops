# script to download the file
import requests
import os
import config
import time
import yaml

# create the directory even if it exists
os.makedirs(config.EXTERNAL_DATA_DIR,exist_ok=True)

DOWNLOAD_PARAMS = config.get_parameters('data_download')

def download_data()-> None:
    logger = config.create_logger(file_name=os.path.basename(__file__))    
    global DOWNLOAD_PARAMS
    response = requests.get(DOWNLOAD_PARAMS.get('url'))
    if response.status_code == 200:
        with open(config.EXTERNAL_DATA_DIR / "tweet_emotions.csv", "wb") as file:
            file.write(response.content)
            logger.debug("CSV file downloaded successfully.")
    else:
        response.raise_for_status()

def main():
    download_data()

if __name__ == "__main__":
    main()

    