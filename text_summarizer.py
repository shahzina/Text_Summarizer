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
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

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

'''
	input:- sentences
	for words in sentence, if word not in stop_words,
	add word to the list.
	return:- words not in stop_words. (stop_words are removed)
'''

def remove_stopwords(s):
	new_s = " ".join([i for i in s if i not in stop_words])

	return new_s
	


    

### CREATE SENTENCE VECTORS
def vectorize_sentences(clean_sentences):
	sentence_vectors = []

	for i in clean_sentences:
		if len(i) !=0:
			v = sum([vectorized_words.get(w, np.zeros((100, ))) for w in i.split()])/(len(i.split()) + 0.001)

		else:
			v = np.zeros((100, ))

		sentence_vectors.append(v)

	return sentence_vectors

### Similarity Matrix
# sim_mat = np.zeros([len(sentences), len(sentences)])


# for i in range(len(sentences)):
#   for j in range(len(sentences)):
#     if i != j:
#       sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


# nx_graph = nx.from_numpy_array(sim_mat)
# scores = nx.pagerank(nx_graph)

# ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# # Extract top 10 sentences as the summary
# for i in range(10):
#   print(ranked_sentences[i][1])




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


	sentence_vecs = vectorize_sentences(clean_sentences)

	sim_mat = np.zeros([len(sentences), len(sentences)])
	for i in range(len(sentences)):
		for j in range(len(sentences)):
			if i != j:
				sim_mat[i][j] = cosine_similarity(sentence_vecs[i].reshape(1,100), sentence_vecs[j].reshape(1,100))[0,0]

	nx_graph = nx.from_numpy_array(sim_mat)
	scores = nx.pagerank(nx_graph)

	ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


	# Extract top 10 sentences as the summary
	for i in range(10):
		print(ranked_sentences[i][1])


"""
OUTPUT

“I was on a nice trajectory then,” Reid recalled.“If I hadn’t got sick, I think I could have started pushing towards the second week at the slams and then who knows.” During a comeback attempt some five years later, Reid added Bernard Tomic and 2018 US Open Federer slayer John Millman to his list of career scalps.
Major players feel that a big event in late November combined with one in January before the Australian Open will mean too much tennis and too little rest.
So I'm not the one to strike up a conversation about the weather and know that in the next few minutes I have to go and try to win a tennis match.
Speaking at the Swiss Indoors tournament where he will play in Sunday’s final against Romanian qualifier Marius Copil, the world number three said that given the impossibly short time frame to make a decision, he opted out of any commitment.
Currently in ninth place, Nishikori with a win could move to within 125 points of the cut for the eight-man event in London next month.
Exhausted after spending half his round deep in the bushes searching for my ball, as well as those of two other golfers he’d never met before, our incredibly giving designated driver asked if we didn’t mind going straight home after signing off so he could rest up a little before heading to work.
“I felt like the best weeks that I had to get to know players when I was playing were the Fed Cup weeks or the Olympic weeks, not necessarily during the tournaments.
“I just felt like it really kind of changed where people were a little bit, definitely in the '90s, a lot more quiet, into themselves, and then it started to become better.” Meanwhile, Federer is hoping he can improve his service game as he hunts his ninth Swiss Indoors title this week.
The former Wimbledon junior champion was full of hope, excited about getting his life back together after a troubled few years and a touch-and-go battle with pancreatitis.
He used his first break point to close out the first set before going up 3-0 in the second and wrapping up the win on his first match point.

"""


