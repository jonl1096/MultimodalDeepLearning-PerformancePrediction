import csv
import sys
csv.field_size_limit(sys.maxsize)
import numpy as np

modenames = ["articles",]
encoder1layers = [([10],[]),]
encoder2layers = [([10],[]),]
encode_sharedrep_layers = ([10],[])
modeidxs = [3,]
labelsidx = 2
dateidxs = (0,1)
datafile = "Data/data_preprocessed_10w_300d.csv"
outputfiletrain = "Data/shared_rep_train.csv"
outputfiletest = "Data/shared_rep_test.csv"


dates_train = []
dates_test = []
vectors_by_mode = [[[],[]] for i in range(len(modeidxs))]
train_or_test = 0
with open(datafile, 'r') as data:
	datareader = csv.reader(data)
	for i,row in enumerate(datareader):
		year = int(str(row[dateidxs[0]])[:4])
		if True:#year < 2016:
			train_or_test = 0
			dates_train.append([str(row[dateidxs[0]]),str(row[dateidxs[1]])])
		else:
			train_or_test = 1
			dates_test.append([str(row[dateidxs[0]]),str(row[dateidxs[1]])])
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
	if len(vectors[0]) > 1:
		vectors_by_mode[i][0] = encoder1[i].predict(vectors[0])
	if len(vectors[1]) > 1:
		vectors_by_mode[i][1] = encoder1[i].predict(vectors[1])

combined_vectors_train = np.copy(vectors_by_mode[0][0])
combined_vectors_test = np.copy(vectors_by_mode[0][1])
for i,vectors in enumerate(vectors_by_mode):
	if i == 0: continue
	combined_vectors_train = np.append(combined_vectors_train, vectors_by_mode[i][0], axis=1)
	combined_vectors_test = np.append(combined_vectors_test, vectors_by_mode[i][1], axis=1)


print("getting individual mode encoders for multi-mode reconstruction")
#optionally add additional encoder layer that can be used to recreate both modes from one mode
encoder2 = [0 for i in range(len(vectors_by_mode))]
for i,vectors in enumerate(vectors_by_mode):
	encoder2[i] = get_encoder(vectors[0], combined_vectors_train, encoder2layers[i][0], encoder2layers[i][1])
	if len(vectors[0]) > 1:
		vectors_by_mode[i][0] = encoder2[i].predict(vectors[0])
	if len(vectors[1]) > 1:
		vectors_by_mode[i][1] = encoder2[i].predict(vectors[1])

combined_vectors_train = np.copy(vectors_by_mode[0][0])
combined_vectors_test = np.copy(vectors_by_mode[0][1])
for i,vectors in enumerate(vectors_by_mode):
	if i == 0: continue
	combined_vectors_train = np.append(combined_vectors_train, vectors_by_mode[i][0], axis=1)
	combined_vectors_test = np.append(combined_vectors_test, vectors_by_mode[i][1], axis=1)

print("getting shared rep encoders")
#get shared rep encoder
shared_rep_encoder = get_encoder(combined_vectors_train,combined_vectors_train,encode_sharedrep_layers[0],encode_sharedrep_layers[1])

print("getting encodings")
#get encodings
if len(combined_vectors_train) > 1:
	encoded_shared_rep_train = shared_rep_encoder.predict(combined_vectors_train)
if len(combined_vectors_test) > 1:
	encoded_shared_rep_test = shared_rep_encoder.predict(combined_vectors_test)

print("done")
print("putting shared rep in file")
with open(outputfiletest, "w") as test_file, open(outputfiletrain, "w") as train_file:
	train_file_writer = csv.writer(train_file)
	test_file_writer = csv.writer(test_file)
	if len(combined_vectors_train) > 1:
		train_encodings = encoded_shared_rep_train.tolist()
		for i,shared_rep in enumerate(train_encodings):
			train_file_writer.writerow([dates_train[i], shared_rep])
	if len(combined_vectors_test) > 1:
		test_encodings = encoded_shared_rep_test.tolist()
		for i,shared_rep in enumerate(test_encodings):
			test_file_writer.writerow([dates_test[i], shared_rep])
#put this into rnn

