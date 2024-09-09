import numpy as np
import pandas as pd
import config
import os
import logging
from sklearn.model_selection import train_test_split


# getting the logger
logger = config.create_logger(file_name = os.path.basename(__file__))

 # getting the parameters
DATASET_PARAMS:dict = config.get_parameters('create_datasets')


def read_data(file_path:os.PathLike) -> pd.DataFrame:
    logger.debug("main data retreived")
    return pd.read_csv(config.EXTERNAL_DATA_DIR / "tweet_emotions.csv")


def preprocess_data(main_data:pd.DataFrame) -> pd.DataFrame:
    # reading the main external data
    logger.debug("Processing raw data")
    # removing the unwanted columns
    main_data.drop(columns=['tweet_id'],inplace=True)

    # getting only the rows where sentiment is happiness or sad
    final_data = main_data[main_data.sentiment.isin(["happiness","sadness"])]

    final_data.loc[:, 'sentiment'] = final_data['sentiment'].infer_objects(copy=False).replace({"happiness": 1, "sadness": 0})

    return final_data


def save_data(df:pd.DataFrame)->None:
    global DATASET_PARAMS
    # do a train_test_split to the data
    os.makedirs(config.RAW_DATA_DIR,exist_ok=True)
    train_data, test_data = train_test_split(df, test_size=DATASET_PARAMS.get('test_size'), random_state=DATASET_PARAMS.get('random_state'))

    logger.debug(f"saving test and train data to location {config.RAW_DATA_DIR}")
    train_data.to_csv(config.RAW_DATA_DIR / "train.csv",index=False)
    test_data.to_csv(config.RAW_DATA_DIR / "test.csv",index=False)


def main():
    main_data = read_data(file_path = config.EXTERNAL_DATA_DIR / "tweet_emotions.csv") 

    final_data = preprocess_data(main_data)

    save_data(final_data)

if __name__ == "__main__":
    main()