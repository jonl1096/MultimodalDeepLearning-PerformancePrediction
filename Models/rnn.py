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
    menu = 'Select a data to perform classification:\n'\
           '- Single Modal -\n'\
           '(1) Statistics\n'\
           '(2) Tweets\n'\
           '(3) News Articles\n'\
           '- Modal concatenation -\n'\
           '(4) Tweets + Statistics\n'\
           '(5) Articles + Statistics\n'\
           '(6) Articles + Tweets\n'\
           '(7) Articles + Tweets + Statistics\n'\
           '- Bimodal Shared Representation -\n'\
           '(8) Tweets & Statistics\n'\
           '(9) Articles & Statistics\n'\
           '(10) Articles & Tweets\n'\
           '- Trimodal Shared Representation -\n'\
           '(11) Articles & Tweets & Statistics\n'\
           '(0) Exit\n'

    while True:
        print(menu)
        option = raw_input("enter option: ")
        # Train, Test, Write Results
        X_train, X_test, y_train, y_test = readCSV(option)
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


def readCSV(option):
    # single modal
    stats_data = '../Data/final_data/statistics.csv'
    tweet_data = '../Data/final_data/tweets_DF.csv'
    artic_data = '../Data/final_data/articles_lda.csv'

    #stats data
    X_stats = pd.DataFrame.from_csv(stats_data, index_col=None)
    Y = X_stats['wins_today']
    X_stats.drop(['previous_date', 'original_date', 'wins_today'], axis=1, inplace=True)
    X_stats = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X_stats))

    #tweet data
    X_tweet = pd.DataFrame.from_csv(tweet_data, index_col=None)

    #article data
    X_artic = pd.DataFrame.from_csv(artic_data, index_col=None, header=None)
    
    # bimodal
    if option == '8':
        bimod_data = '../Data/processed/bimodal_TS.csv'
        X_bimod = pd.DataFrame.from_csv(bimod_data, index_col=None, header=None)
    if option == '9':
        bimod_data = '../Data/processed/bimodal_AS.csv'
        X_bimod = pd.DataFrame.from_csv(bimod_data, index_col=None, header=None)
    if option == '10':
        bimod_data = '../Data/processed/bimodal_AT_deep_with.csv'
        X_bimod = pd.DataFrame.from_csv(bimod_data, index_col=None, header=None)

    # trimodal
    trimod_data = '../Data/processed/trimodal_with.csv'
    X_trimod = pd.DataFrame.from_csv(trimod_data, index_col=None, header=None) 
    
    #a, b, c = 0.001, 10, 5
    #X_stats, X_tweet, X_artic = a*X_stats, b*X_tweet, c*X_artic
    
    if option == '1':
        X = X_stats
    elif option == '2':
        X = X_tweet
    elif option == '3':
        X = X_artic
    elif option == '4':
        X = pd.concat([X_tweet, X_stats], axis=1)   
    elif option == '5':
        X = pd.concat([X_artic, X_stats], axis=1)   
    elif option == '6':
        X = pd.concat([X_artic, X_tweet], axis=1)   
    elif option == '7':
        X = pd.concat([X_tweet, X_artic, X_stats], axis=1)   
    elif option == '8' or option == '9' or option == '10':
        #X_bimod = pd.DataFrame.from_csv("../Data/processed/transformed_shared_rep_total_data.csv", index_col=None, header=None)
        X = X_bimod
    elif option == '11':
        X = X_trimod
    elif option == '0':
        exit(1)
    else:
        print("wrong option.")
        exit(1)

    n = 371
    X_train, X_test, y_train, y_test = X[:n], X[n:], Y[:n], Y[n:]    # single stats
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
	main()


