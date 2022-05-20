
#get the data
#!wget https://archive.org/download/stackexchange/cs.stackexchange.com.7z/Posts.xml

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def pre_process(text):
	'''Preprocess input text'''

	text=text.lower()
	text=re.sub("</?.*?>"," <> ",text) #remove html tags
	text=re.sub("(\\d|\\W)+"," ",text) # remove special characters and digits
	text = text.strip() #remove blank characters
	return text


# Load the dataset
data = pd.read_xml("Posts.xml")
data = data.Body #we are taking on the text body
data.fillna('', inplace=True)
data = data.apply(lambda x: pre_process(x))

# Compute TF-IDF
tfidf_vectorizer = TfidfVectorizer(use_idf=True,max_df=0.7)
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(data)


# Store the TF-IDF values for later use
with open('tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f, pickle.HIGHEST_PROTOCOL)