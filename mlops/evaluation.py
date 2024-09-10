# this is the prediction part
import config
import pickle
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import classification_report
logger = config.create_logger(file_name=os.path.basename(__file__))
# loading the model
try:

    with open(config.MODELS_DIR / "gbc_model.pkl",'rb') as model:
        gbc_model = pickle.load(model)
except Exception as e:
    print(e)
    

def model_evaluation():

    # loading the test data
    X_test =  config.read_data(config.INTERIM_DATA_DIR / "test_transformed.csv")
    assert not X_test.empty , "The test file is empty"


    logger.debug("Predicting the Values")
    predictions = gbc_model.predict(X_test.iloc[:,0:-1].values)
    y_true = np.array(X_test.iloc[:,-1].values)


    logger.debug("saving metrics")
    # Save classification report to metrics.json
    with open(config.REPORTS_DIR / "metrics_gbc.json", 'w') as file:
        json.dump(classification_report(y_true, predictions), file)


def main():
    model_evaluation()

if __name__ == "__main__":
    main()