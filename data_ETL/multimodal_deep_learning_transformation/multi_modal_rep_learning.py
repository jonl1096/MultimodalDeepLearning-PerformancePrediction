import csv
import sys
csv.field_size_limit(sys.maxsize)
import numpy as np

menu = 'Choose between the following encoding options:\n'\
	   '(1) Bimodal shared representation learning\n'\
	   '(2) Bimodal shared representation learning with labels\n'\
	   '(3) Trimodal shared representation learning\n'\
	   '(4) Trimodal shared representation learning with labels\n'

print(menu)

option = raw_input("Enter option: ")

# parameters
bimodal = False
withlabel = False
if option == '1' or option == '2':
	bimodal = True
if option == '2' or option == '4':
	withlabel = True

# bimodal
if bimodal:
	modenames = ["articles","tweets","stats"]
	modeidxs = [1,5]
	encoder1layers = [([20],[]),([20],[]),]
	encoder2layers = [([10],[]),([10],[]),]
	encode_sharedrep_layers = ([10],[])
else:
	modenames = ["articles","tweets","stats"]
	modeidxs = [0,1,5]
	encoder1layers = [([20],[]),([20],[]),([20],[]),]
	encoder2layers = [([10],[]),([10],[]),([10],[]),]
	encode_sharedrep_layers = ([10],[])

datafile = "combined_final.csv"
outputfiletrain = "shared_rep_train_data.csv"
outputfiletest = "shared_rep_test_data.csv"
outputfiletotal = "shared_rep_total_data.csv"

labelsidx = 4
dateidxs = (2,3)
#datafile = "Data/data_preprocessed_10w_300d.csv"
# outputfiletrain = "Data/shared_rep_train_LDA.csv"
# outputfiletest = "Data/shared_rep_test_LDA.csv"
# outputfiletotal = "Multimodal/shared_rep_trimodal_LDA_deeper.csv"


dates_train = []
dates_test = []
train_labels = []
test_labels = []
vectors_by_mode = [[[],[]] for i in range(len(modeidxs))]
train_or_test = 0
with open(datafile, 'r') as data:
	datareader = csv.reader(data)
	next(datareader)
	for i,row in enumerate(datareader):
		year = int(str(row[dateidxs[0]])[-2:])
		if year < 2016:
			train_or_test = 0
			dates_train.append([str(row[dateidxs[0]]),str(row[dateidxs[1]])])
			train_labels.append([int(row[labelsidx])])
		else:
			train_or_test = 1
			dates_test.append([str(row[dateidxs[0]]),str(row[dateidxs[1]])])
			test_labels.append([int(row[labelsidx])])
		for i, modeidx in enumerate(modeidxs):
			#print(len(vectors_by_mode), len(row))
			vectors_by_mode[i][train_or_test].append(eval(row[modeidx]))


#put data in numpy matricies
for i,vectors in enumerate(vectors_by_mode):
	vectors_by_mode[i][0] = np.matrix(vectors[0])
	vectors_by_mode[i][1] = np.matrix(vectors[1])
train_labels = np.matrix(train_labels)
test_labels = np.matrix(test_labels)

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
	for i,dim in enumerate(decoding_dim_list):
		if i == 0:
			decoded = Dense(dim)(encoded)
		else:
			decoded = Dense(dim)(decoded)
	if len(decoding_dim_list) > 0:
		decoded = Dense(vec_len_out)(decoded)
	else:
		decoded = Dense(vec_len_out)(encoded)

	autoencoder = Model(input_vector, decoded)

	encoder = Model(input_vector, encoded)

	# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	autoencoder.fit(input_vectors, output_vectors,
					epochs=100)

	return encoder

print("getting individual mode encoders")
#get individual mode encoders
encoder1 = [0 for i in range(len(vectors_by_mode))]
for i,vectors in enumerate(vectors_by_mode):
	print(vectors[0].shape, train_labels.shape)
	if withlabel:
		encoder1[i] = get_encoder(vectors[0], np.append(vectors[0], train_labels, axis=1), encoder1layers[i][0], encoder1layers[i][1])
	else:
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
	if withlabel:
		encoder2[i] = get_encoder(vectors[0], np.append(combined_vectors_train, train_labels, axis=1), encoder2layers[i][0], encoder2layers[i][1])
	else:
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
if withlabel:
	shared_rep_encoder = get_encoder(combined_vectors_train,np.append(combined_vectors_train, train_labels, axis=1),encode_sharedrep_layers[0],encode_sharedrep_layers[1])
else:
	shared_rep_encoder = get_encoder(combined_vectors_train,combined_vectors_train,encode_sharedrep_layers[0],encode_sharedrep_layers[1])

print("getting encodings")
#get encodings
if len(combined_vectors_train) > 1:
	encoded_shared_rep_train = shared_rep_encoder.predict(combined_vectors_train)
if len(combined_vectors_test) > 1:
	encoded_shared_rep_test = shared_rep_encoder.predict(combined_vectors_test)

print("done")
print("putting shared rep in file")
with open(outputfiletest, "w") as test_file, open(outputfiletrain, "w") as train_file, open(outputfiletotal, "w") as total_file:
	train_file_writer = csv.writer(train_file)
	test_file_writer = csv.writer(test_file)
	total_file_writer = csv.writer(total_file)
	if len(combined_vectors_train) > 1:
		train_encodings = encoded_shared_rep_train.tolist()
		for i,shared_rep in enumerate(train_encodings):
			train_file_writer.writerow([dates_train[i], shared_rep])
			total_file_writer.writerow([dates_train[i], shared_rep])
	if len(combined_vectors_test) > 1:
		test_encodings = encoded_shared_rep_test.tolist()
		for i,shared_rep in enumerate(test_encodings):
			test_file_writer.writerow([dates_test[i], shared_rep])
			total_file_writer.writerow([dates_test[i], shared_rep])
#put this into rnn

