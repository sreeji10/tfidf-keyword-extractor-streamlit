import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


#Setting up Streamlit configurations
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title('Keyword extraction using TFIDF')
st.markdown('Display top 5 keywords')
Text = st.text_input('Enter the sentence & press enter key')


@st.cache(allow_output_mutation=True)
def load():
	''' Load the calculated TFIDF weights'''
	data = None
	pkl_path = Path(__file__).parents[1] / 'data/tfidf.pickle'
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
	return data


def pre_process(text):
	'''Preprocess input text'''

	text=text.lower()
	text=re.sub("</?.*?>"," <> ",text) #remove html tags
	text=re.sub("(\\d|\\W)+"," ",text) # remove special characters and digits
	text = text.strip() #remove blank characters
	return text


def process(tfidf_vectorizer, text):
	'''Compute the Top 5 TFIDF scores'''

	if text is not None and text != '' and text != ' ':
		txt = tfidf_vectorizer.transform([text])
		df = pd.DataFrame(txt.T.todense(), index=tfidf_vectorizer.get_feature_names_out(), columns=["tfidf"])
		print(df)
		if len(text) >= 5:
			df = df.sort_values(by=["tfidf"],ascending=False)[:5]
		else:
			df = df.sort_values(by=["tfidf"],ascending=False)[:len(text)]
		return df
	return ''


def run():
	tfidf_vectorizer = load()
	text = pre_process(Text)
	val = process(tfidf_vectorizer, text)
	print(val, Text)
	st.write(val)

if __name__ == '__main__':
	run()