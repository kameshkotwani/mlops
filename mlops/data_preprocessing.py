from aiohttp import TraceConnectionQueuedEndParams
import config
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# read the train and test csv file
train_raw = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
assert not train_raw.empty, "train_raw does not have any data"

test_raw = pd.read_csv(config.RAW_DATA_DIR / "test.csv")
assert not test_raw.empty, "test_raw does not have any data"

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.content.iloc[i].split()) < 3:
            df.content.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df


# creating processed dir
os.makedirs(config.PROCESSED_DATA_DIR)


print("Processing the datasets train and test")
train_data = normalize_text(train_raw)
train_data.fillna('empty',inplace=True)

print("saving train_processed.csv")
train_data.to_csv(config.PROCESSED_DATA_DIR / "train_processed.csv",index=False)

test_data = normalize_text(test_raw)

test_data.fillna('empty',inplace=True)
print("saving test_processed.csv")
test_data.to_csv(config.PROCESSED_DATA_DIR / "test_processed.csv",index=False)
