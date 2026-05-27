import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # hapus url
    text = re.sub(r'http\S+', '', text)

    # hapus angka
    text = re.sub(r'\d+', '', text)

    # hapus punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenizing
    tokens = word_tokenize(text)

    # stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # gabung kembali
    text = ' '.join(tokens)

    return text


def preprocess_dataset(df):

    # hapus missing value
    df = df.dropna()

    # hapus duplicate
    df = df.drop_duplicates()

    # preprocessing text
    df['clean_text'] = df['text'].astype(str).apply(preprocess_text)

    return df
  