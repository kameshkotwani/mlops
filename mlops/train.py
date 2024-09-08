from config import INTERIM_DATA_DIR, MODELS_DIR
import pandas as pd
import pickle
train_data = pd.read_csv(INTERIM_DATA_DIR / "train_transformed.csv")

from sklearn.ensemble import GradientBoostingClassifier


# creating X and y for training
X = train_data.iloc[:,0:-1].values
y = train_data.iloc[:,-1].values

# Define and train the XGBoost model
gbc = GradientBoostingClassifier(n_estimators=50)
gbc.fit(X, y)

# saving_model
pickle.dump(gbc, open(MODELS_DIR / "gbc_model.pkl",'wb'))




