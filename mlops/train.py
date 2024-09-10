import config 
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier

# getting parameters
TRAIN_PARAMS = config.get_parameters('train')

def train_and_save_model()->None:
    global TRAIN_PARAMS
    train_data = config.read_data(config.INTERIM_DATA_DIR / "train_transformed.csv")
    
    logger = config.create_logger(file_name=os.path.basename(__file__))
    logger.debug("Training Model")
    X = train_data.iloc[:,0:-1].values
    y = train_data.iloc[:,-1].values

    # Define and train the XGBoost model
    gbc = GradientBoostingClassifier(n_estimators=TRAIN_PARAMS.get('gbc_n_estimators'))
    gbc.fit(X, y)

    logger.debug(f"saving model at path {config.MODELS_DIR} / gbc_model.pkl")
    # saving_model
    pickle.dump(gbc, open(config.MODELS_DIR / "gbc_model.pkl",'wb'))



def main():
    train_and_save_model()

if __name__ == "__main__":
    main()
