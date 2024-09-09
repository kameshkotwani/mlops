import numpy as np
import pandas as pd
import config
import os
from sklearn.model_selection import train_test_split

# getting the parameters
DATASET_PARAMS:dict = config.get_parameters('create_datasets')

os.makedirs(config.RAW_DATA_DIR,exist_ok=True)
# reading the main external data
print("Processing raw data")
main_data = pd.read_csv(config.EXTERNAL_DATA_DIR / "tweet_emotions.csv")

# removing the unwanted columns
main_data.drop(columns=['tweet_id'],inplace=True)

# getting only the rows where sentiment is happiness or sad
final_data = main_data[main_data.sentiment.isin(["happiness","sadness"])]

# doing binary encoding to the data
# Need to check what the hell is going on here
# final_data['sentiment'] = final_data['sentiment'].replace({"happiness":1,"sadness":0})

# new version
# Replace sentiment values and convert to int
final_data.loc[:, 'sentiment'] = final_data['sentiment'].infer_objects(copy=False).replace({"happiness": 1, "sadness": 0})




# do a train_test_split to the data
train_data, test_data = train_test_split(final_data, test_size=DATASET_PARAMS.get('test_size'), random_state=DATASET_PARAMS.get('random_state'))

train_data.to_csv(config.RAW_DATA_DIR / "train.csv",index=False)
test_data.to_csv(config.RAW_DATA_DIR / "test.csv",index=False)