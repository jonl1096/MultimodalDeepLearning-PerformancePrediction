# LSTM for sequence classification in the IMDB dataset
import pandas as pd
import numpy as np
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import preprocessing


# fix random seed for reproducibility
#np.random.seed(7)

def main():
    team_name = "Orioles"
    model_names = ['svm', 'nn', 'rf']

    # Train, Test, Write Results
    X_train, X_test, y_train, y_test = readCSV()
    run_rnn(X_train, X_test, y_train, y_test)



def run_rnn(X_train, X_test, y_train, y_test):
	x_len = X_train.shape[1]
	X_train = X_train.values
	X_test = X_test.values

	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	X_train = np.reshape(X_train, (X_train.shape[0], 1, x_len))
	X_test = np.reshape(X_test, (X_test.shape[0], 1, x_len))

	model = Sequential()
	model.add(Dense(50, input_shape=(1, x_len)))
	model.add(Dropout(0.2))
	model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))#, input_shape=(1, x_len)))
	model.add(Dropout(0.3))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, nb_epoch=100, batch_size=32)
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))


def readCSV():
    stats_data = '../full_data/final_data/statistics.csv'
    tweet_data = '../full_data/final_data/tweets_DF.csv'
    artic_data = '../full_data/final_data/articles_tf.csv'
    bimod_data = '../full_data/final_data/bimodal_TnA.csv'

    #stats data
    X_stats = pd.DataFrame.from_csv(stats_data, index_col=None)
    Y = X_stats['wins_today']
    X_stats.drop(['previous_date', 'original_date', 'wins_today'], axis=1, inplace=True)
    X_stats = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X_stats))

    #tweet data
    X_tweet = pd.DataFrame.from_csv(tweet_data, index_col=None)

    #article data
    X_artic = pd.DataFrame.from_csv(artic_data, index_col=None, header=None)

    #bimodal data
    X_bimod = pd.DataFrame.from_csv(bimod_data, index_col=None, header=None)
    #a, b, c = 0.001, 10, 5
    #X_stats, X_tweet, X_artic = a*X_stats, b*X_tweet, c*X_artic
    X = X_bimod
    #X = pd.concat([X_stats, X_tweet, X_artic], axis=1)

    # single stats
    x_len = X.shape[0]
    cut = 0.5
    n = int(x_len * cut)
    X_train, X_test, y_train, y_test = X[:n], X[n:], Y[:n], Y[:n]
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=42)
    return X_train, X_test, y_train, y_test




if __name__ == '__main__':
	main()


