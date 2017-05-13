import csv
import sys
csv.field_size_limit(sys.maxsize)
import numpy as np

dates = []
stats_vectors = []
article_vectors = []
twitter_vectors = []
datafile = "Data/data_preprocessed.csv"
outputfile = "Data/shared_rep.csv"
with open(datafile, 'r') as data:
	datareader = csv.reader(data)
	for row in datareader:
		dates.append(str(row[0]))
		stats_vectors.append(eval(row[1]))
		article_vectors.append(eval(row[2]))
		# article_vectors[i] = eval(row[3])
print(stats_vectors[0])
print(article_vectors[0])
# print(twitter_vectors[0])

#put data in numpy matricies
stats_vectors = np.matrix(stats_vectors)
article_vectors = np.matrix(article_vectors)
# twitter_vectors = np.matrix(twitter_vectors)

print("importing keras modules")
from keras.layers import Input, Dense
from keras.models import Model
print("keras successfully imported!")

def get_encoder(input_vectors, output_vectors, encoding_dim_list, decoding_dim_list):
	vec_len_in = input_vectors.shape[1]
	vec_len_out = output_vectors.shape[1]
	print(vec_len_in,vec_len_out)
	input_vector = Input(shape=(vec_len_in,))
	#encoding layer(s)
	encoded = Dense(encoding_dim_list[0])(input_vector)
	for dim in encoding_dim_list[1:]:
		encoded = Dense(dim)(encoded)

	#decoding layer(s)
	for dim in decoding_dim_list:
		if i == 0:
			decoded = Dense(dim)(encoded)
		else:
			decoded = Dense(dim)(decoded)
	decoded = Dense(vec_len_out)(encoded)

	autoencoder = Model(input_vector, decoded)

	encoder = Model(input_vector, encoded)

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	autoencoder.fit(input_vectors, output_vectors)

	return encoder

print("getting individual mode encoders")
#get individual mode encoders
stats_encoder1 = get_encoder(stats_vectors,stats_vectors, [10], [])
article_encoder1 = get_encoder(article_vectors,article_vectors, [10], [])

print("getting encodings")
#get encodings
encoded_stats = stats_encoder1.predict(stats_vectors)
encoded_articles = article_encoder1.predict(article_vectors)
combined_vector = np.append(encoded_stats, encoded_articles, axis=1)

print("getting individual mode encoders for multi-mode reconstruction")
#optionally add additional encoder layer that can be used to recreate both modes from one mode
stats_encoder2 = get_encoder(encoded_stats,combined_vector,[10],[])
article_encoder2 = get_encoder(encoded_articles,combined_vector,[10],[])

print("getting encodings")
#get encodings
encoded_stats = stats_encoder2.predict(encoded_stats)
encoded_articles = article_encoder2.predict(encoded_articles)
combined_vector = np.append(encoded_stats, encoded_articles, axis=1)

print("getting shared rep encoders")
#get shared rep encoder
shared_rep_encoder = get_encoder(combined_vector,combined_vector,[10],[])

print("getting encodings")
#get encodings
encoded_shared_rep = shared_rep_encoder.predict(combined_vector)

print("done")
print("putting shared rep in file")
with open(outputfile, "w") as outputfile:
	sharedrepwriter = csv.writer(outputfile)
	sharedreplist = encoded_shared_rep.tolist()
	for i,shared_rep in enumerate(sharedreplist):
		sharedrepwriter.writerow([dates[i], shared_rep])
#put this into rnn

