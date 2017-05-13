import csv
import sys
csv.field_size_limit(sys.maxsize)
import numpy as np

modenames = ["articles",]
encoder1layers = [([10],[]),]
encoder2layers = [([10],[]),]
modeidxs = [3,]
labelsidx = 2
dateidxs = (0,1)
datafile = "Data/data_preprocessed_10w_300d.csv"
outputfile = "Data/shared_rep.csv"


dates = []
vectors_by_mode = [[[],[]] for i in range(len(modeidxs))]
train_or_test = 0
with open(datafile, 'r') as data:
	datareader = csv.reader(data)
	for i,row in enumerate(datareader):
		dates.append([str(row[dateidxs[0]]),str(row[dateidxs[1]])])
		year = int(dates[len(dates)-1][0][:4])
		if year < 2016:
			train_or_test = 0
		else:
			train_or_test = 1
		for i, modeidx in enumerate(modeidxs):
			vectors_by_mode[i][train_or_test].append(eval(row[modeidx]))


#put data in numpy matricies
for i,vectors in enumerate(vectors_by_mode):
	vectors_by_mode[i][0] = np.matrix(vectors[0])
	vectors_by_mode[i][1] = np.matrix(vectors[1])

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
encoder1 = [0 for i in range(len(vectors_by_mode))]
for i,vectors in enumerate(vectors_by_mode):
	encoder1[i] = get_encoder(vectors[0], vectors[0], encoder1layers[i][0], encoder1layers[i][1])
	vectors[0] = encoder1[i].predict(vectors[0])
	vectors[1] = encoder1[i].predict(vectors[1])

combined_vector = np.zeros()

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

