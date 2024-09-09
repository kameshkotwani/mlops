import numpy as np
import pandas as pd
import os
import config

from sklearn.feature_extraction.text import CountVectorizer

# get the parameters dict
FE_PARAMS = config.get_parameters('feature_engineering')

# getting the files from processed dir
train_data = pd.read_csv(config.PROCESSED_DATA_DIR / "train_processed.csv")
test_data = pd.read_csv(config.PROCESSED_DATA_DIR / "test_processed.csv")
train_data.fillna("",inplace=True)
test_data.fillna("",inplace=True)

print("creating x and y")
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

print("applying transformations")
# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=FE_PARAMS.get('cv_max_features'))

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)


train_df = pd.DataFrame(X_train_bow.toarray())
test_df = pd.DataFrame(X_test_bow.toarray())

# creating the final dataframes
train_df['label'] = y_train
test_df['label'] = y_test


# saving the data to interim directoy
os.makedirs(config.INTERIM_DATA_DIR,exist_ok=True)
train_df.to_csv(config.INTERIM_DATA_DIR / "train_transformed.csv",index=False)
test_df.to_csv(config.INTERIM_DATA_DIR / "test_transformed.csv",index=False)