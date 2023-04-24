import markovify, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import re

# Load the dataset about tweets
cs_ds = pd.read_csv('preprocessed_tweets_1mill.csv')

cs_ds['tweet'] = cs_ds['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))

#Splicing our dataset so that it isn't too big
cs_ds = cs_ds.sample(n=400000, random_state=42)

# Split the dataset into training and test sets
train_ds, test_ds = train_test_split(cs_ds, test_size=0.2, random_state=42)

# Tokenize our tweet column and split up each word
train_ds['sentences'] = train_ds['tweet'].apply(sent_tokenize)
test_ds['sentences'] = test_ds['tweet'].apply(sent_tokenize)

train_sentences = [sentence for sentences in train_ds['sentences'] for sentence in sentences]
train_text = '\n'.join(train_sentences)

model_2 = markovify.NewlineText(train_text, state_size=2)

with open('markov_trained_model.pkl', 'wb') as file:
    pickle.dump(model_2, file)
