import os.path
import csv
import sys
csv.field_size_limit(sys.maxsize)
import numpy as np


data_file = "Data/data14_16.csv"
article_preprocess = "word_embeddings"
twitter_preprocess = "word_embeddings"


def get_data(datafilename):
	data = []
	with open(datafilename, 'r') as datafile:
		datareader = csv.reader(datafile)
		for row in datareader:
			data.append(row)
	return data


def get_embeddings():
	glovefile = "glove/glove.6B.50d.txt"
	embeddings = {}
	with open(glovefile, 'r') as embeddings_file:
		for line in embeddings_file:
			linesplit = line.split()
			word = linesplit[0]
			values = [float(numstr) for numstr in linesplit[1:]]
			embeddings[word] = values
	print("done getting embeddings!")

	return embeddings

def get_embedding_rep(words, embeddings):
	words = words.split() #[:10]
	final_vector = np.zeros(len(embeddings[embeddings.keys()[0]]))
	num_embedded_words = 0
	for i,word in enumerate(words):
		if word in embeddings.keys():
			final_vector = np.add(final_vector, np.array(embeddings[word]))
			num_embedded_words += 1
		else:
			print(word)
		print(str(i)+" / "+str(len(words)), word)
	final_vector /= num_embedded_words

	return list(final_vector)

data = get_data(data_file)
embeddings = get_embeddings()
#Preprocessing
#0 - stats, 1 - articles, 2 - tweets

#for date, stats, and articles
with open('Data/data_preprocessed.csv', 'w') as data_preprocessed:
	datawriter = csv.writer(data_preprocessed)
	for example in data:
		print(example[0])
		stats = [float(stat) for stat in eval(example[1])]
		#get article rep
		if article_preprocess == "word_embeddings":
			datawriter.writerow([example[0], stats, get_embedding_rep(example[2], embeddings)])
		# elif article_preprocess == "LDA":
		# 	pass
		else:
			raise Exception("Impropper article preprocessing specification!")

#concat on twitter preprocessed data

print("done preprocessing")

