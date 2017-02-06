# Similar to the text-clustering problem now you will load
# some files from wikipidia and try to group them
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob


def ReadFile(string):
	with open('./'+string+'.txt','r') as f:
		data = f.read()
	return data

#Loading .txt files from the same directory
textFiles = glob.glob("*.txt")
titles = []
[titles.append(names[:-4]) for names in textFiles]
contents = []
[contents.append(ReadFile(names)) for names in titles] 
# Now with pandas we create a dataset with text and titles
data = {'Title': titles, 'Content' :contents} 
df = pd.DataFrame(data)

# Creating corpus
corpus = [] 
for i in range(df['Content'].size):
	corpus.append(df['Content'][i])

from sklearn.feature_extraction.text import TfidfVectorizer
stopWords = nltk.corpus.stopwords.words('portuguese')

vectorizer = TfidfVectorizer(stop_words = stopWords)
X = vectorizer.fit_transform(corpus)
X.todense()

# Now we can use KMeans to classify the text
k = 5
dist = 1 - cosine_similarity(X)
model = KMeans(n_clusters = k)
model.fit(X)

no_words = 3
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()
labels = model.labels_

print('Top terms per cluster:\n')
for i in range(k):

	print("Cluster %i content:"%i, end='')
	for title in df['Content'][labels == i]:
		print(' %s,' %title, end='')
		print(" ")
	print('\n')
	print("Cluster %i words:"%i, end='')
	print('\n')
	for ind in order_centroids[i,:no_words]:
		print(' %s' % terms[ind],end=','),
	print('\n')
