import config 
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# getting parameters
TRAIN_PARAMS = config.get_parameters('train')


train_data = pd.read_csv(config.INTERIM_DATA_DIR / "train_transformed.csv")



# creating X and y for training
X = train_data.iloc[:,0:-1].values
y = train_data.iloc[:,-1].values

# Define and train the XGBoost model
gbc = GradientBoostingClassifier(n_estimators=TRAIN_PARAMS.get('gbc_n_estimators'))
gbc.fit(X, y)

# saving_model
pickle.dump(gbc, open(config.MODELS_DIR / "gbc_model.pkl",'wb'))




