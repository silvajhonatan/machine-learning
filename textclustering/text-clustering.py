# If we have a set of texts and we want to label them 
# by content what can we do ? 
# First thing you can do is look at the vocabulary of the overall
# texts and with them you can create what they call bag of words
# if we have 'I Love bacon' , 'This is a good movie', 'Machine learning
# is cool' , 'I like potatos' and we want to look label, what we might get
# is Love, bacon, good, movie, machine, learning, cool, potatos 
# so if a new phrase like 'I very much like potatos, but I do like bacon'
# what we want to do is transform this into a numpy array such as 
# [0,1,0,0,0,0,0,1] as we have a set of content ['love','bacon',....]
# and we might also divide by the total amount of words in the lexicon
# [0,1/2,0,0,0,0,0,1/2]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = ['I love bacon', 'This was an awesome movie', "I saw a great movie just last night", 
		  'I like potatos']

count_vect = CountVectorizer(stop_words = 'english')
Z = count_vect.fit_transform(corpus)
Z.todense()
vocab = count_vect.get_feature_names()
# We can print the vocabulary of the corpus
print(vocab)
# We can get the frequency matrix
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(corpus)
X.todense()
# Now we can use KMeans to classify the text
k = 2
dist = 1 - cosine_similarity(X)
model = KMeans(n_clusters = k)
model.fit(X)

print('Top terms per cluster:\n')
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
	print("Cluster %i:"%i, end='')
	for ind in order_centroids[i,:4]:
		print(' %s' % terms[ind],end='')
	print("")
