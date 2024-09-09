# this is the prediction part
import config
import pickle
import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report
# loading the model
try:

    with open(config.MODELS_DIR / "gbc_model.pkl",'rb') as model:
        gbc_model = pickle.load(model)
except Exception as e:
    print(e)
    
    
# loading the test data
X_test =  pd.read_csv(config.INTERIM_DATA_DIR / "test_transformed.csv")
assert not X_test.empty , "The test file is empty"


print("Predicting the Values")
predictions = gbc_model.predict(X_test.iloc[:,0:-1].values)
y_true = np.array(X_test.iloc[:,-1].values)

print(classification_report(y_true,predictions,output_dict=True))

# Generate classification report as a dictionary
report_dict = classification_report(y_true, predictions)

# Save classification report to metrics.json
with open(config.REPORTS_DIR / "metrics_gbc.json", 'w') as file:
    json.dump(report_dict, file)


