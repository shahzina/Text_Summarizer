# -*- coding: utf-8 -*-

# GloVe download link- https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db 

"""
Extractive text summarizer using TextRank algorithm on Tennis commentary.
Goal is to generate a summary of a tennis match. 

"""

import numpy as np 
import pandas as pd 
import nltk
# from nltk.corpus import stopwords
#nltk.download('punkt') #one time execution
import re
from nltk.tokenize import sent_tokenize

df = pd.read_csv("tennis_articles.csv", encoding = 'windows-1252')

#print(df.head())
# print(df['article_text'][0])
#print(type(df['article_text']))

###### SENTENCE TOKENISER
def sentence_list(df_column):
	sentences = []
	for s in df_column: #for s in df['article_text']
	  sentences.append(sent_tokenize(s))

	sentences = [y for x in sentences for y in x] # flatten list

	return sentences 

#print(sentence_list(df['article_text']))

def word_vectors(f):
	word_embeddings = {}
	# f = open('word-vectors/glove.6B.100d.txt', encoding='utf-8')

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype = 'float32')
		word_embeddings[word] = coefs
	f.close()

	return word_embeddings

#print(len(word_embeddings)) ### ans: 400000


### TEXT PROCESSING STEP:
# we clean text to get rid of noise. 
# clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
# clean_sentences = [s.lower() for s in clean_sentences]
#nltk.download('stopwords') #### UNCOMMENT when running for first time
stop_words = nltk.corpus.stopwords.words('english') 

def remove_stopwords(s):
	'''
	input:- sentences
	for words in sentence, if word not in stop_words,
	add word to the list.
	return:- words not in stop_words. (stop_words are removed)
	'''
    new_s = " ".join([i for i in s if i not in stop_words])
    return new_s

### CREATE SENTENCE VECTORS






if __name__ == "__main__":
	sentences = sentence_list(df['article_text'])

	#define f
	f = open('word-vectors/glove.6B.100d.txt', encoding='utf-8')
	vectorized_words = word_vectors(f)

	#convert to pandas series, replace given strings with space
	clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
	#convert all to lower case
	clean_sentences = [s.lower() for s in clean_sentences]
	#remove stopwords from clean sentences
	clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


