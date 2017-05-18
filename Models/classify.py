# coding: utf-8
import csv
import pickle
import numpy as np
import pandas as pd
import copy
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


def main():
    team_name = "Orioles"
    model_names = ['svm', 'nn', 'rf']
    clear_txt_files(model_names)

    # Train, Test, Write Results
    train_data, test_data = readCSV()
    scores = []
    for model_name in model_names:
        train(train_data, team_name, model_name, grid_search=True)
        clf = load(team_name)
        scores.append((model_name, predict(clf, test_data, team_name, model_name, save=True), clf))
    wins_nexts = np.unique(test_data[1])
    write_summary(scores, team_name, len(wins_nexts), len(train_data[1]), wins_nexts)
    scores.sort(key=lambda tup:tup[1], reverse=True)
    write_best_scores(scores, team_name, test_data[1])

    # Making predictions    
    #predict_atbat(clf)


def readCSV():
    stats_data = '../Data/final_data/statistics.csv'
    tweet_data = '../Data/final_data/tweets_DF.csv'
    artic_data = '../Data/final_data/articles_lda.csv'
    bimod_data = '../Data/full_data/processed/bimodal_AT_deep_with.csv'
    trimod_data = '../Data/full_data/processed/trimdal_with.csv'

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
    X_trimod = pd.DataFrame.from_csv(bimod_data, index_col=None, header=None) 
    #a, b, c = 0.001, 10, 5
    #X_stats, X_tweet, X_artic = a*X_stats, b*X_tweet, c*X_artic
    
    #X = X_artic
    #X = pd.concat([X_bimod], axis=1)
    X = X_tweet
    n = 371
    X_train, X_test, y_train, y_test = X[:n], X[n:], Y[:n], Y[n:]    # single stats
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=42)
    return [X_train, y_train], [X_test, y_test]


def populateData(X_train, y_train):
    labels = np.unique(y_train)
    labels_counts = []
    for label in labels:
        labels_counts.append((label,np.sum(y_train == label)))
    labels_counts.sort(key=lambda tup:tup[1], reverse=True)
    
    max_counts = labels_counts[0][1]
    #print("Max Counts:%d" %max_counts)

    # sample rest of the smaller labeled datasets and append
    for i in range(1, len(labels_counts)):
        sample_size = max_counts - labels_counts[i][1]
        curr_label = labels_counts[i][0]
        #print("Label %s is %d many short in numbers"%(labels_counts[i][0], max_counts - labels_counts[i][1]))
        indicies = [i for i, x in enumerate(y_train==curr_label) if x]
        sampled_idx = np.random.choice(indicies, sample_size, replace=True)
        X_train = X_train.append(X_train.iloc[sampled_idx], ignore_index=True)
        y_train = y_train.append(pd.Series([curr_label] * sample_size), ignore_index=True)
    
    perm_idx = np.random.permutation(y_train.index)
    X_train = X_train.reindex(perm_idx)
    y_train = y_train.reindex(perm_idx)
    return X_train, y_train


def train(data, team_name, model_name, grid_search):
    X = data[0]
    y = data[1]
    n_samples = len(X)

    if len(np.unique(y)) is 1:
        print(team_name)
        print(np.unique(n_samples))
        print(n_samples)
        exit(1)
#    clf = OneVsRestClassifier(LinearSVC(random_state=0))
#    clf = OneVsOneClassifier(LinearSVC(random_state=0))
    if model_name == 'svm':
        clf = svm.SVC(decision_function_shape='ovr', gamma='auto', kernel='rbf')
    #    clf = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), param_grid)
    elif model_name == 'nn':
        clf = MLPClassifier()
    elif model_name == 'rf':
        clf = RandomForestClassifier()
    
    # grid search
    if grid_search:
        param_grid = get_param_grid(model_name)
        grid_search = GridSearchCV(clf, param_grid=param_grid)#, scoring='f1_macro')
        print("\nfitting..")
        print(param_grid)
        grid_search.fit(X, y)
        print("\nfitting done!")
        clf = grid_search.best_estimator_
    else:
        clf.fit(X, y)

    # save the classifier
    file_name = "../classifiers/" + team_name + ".pkl"
    joblib.dump(clf, file_name)


def load(team_name):
    file_name = "../classifiers/" + team_name + ".pkl"
    clf = joblib.load(file_name)
    return clf


def predict(clf, data, team_name, model_name, save):
    X = data[0]
    y = data[1]
    predictions = clf.predict(X)
    eval(clf, y, predictions)
    if save:
        write_results(y, predictions, team_name, model_name)
    return scores(clf, X, y)


def get_param_grid(model):
    if (model == 'nn'):
      param_grid = {'hidden_layer_sizes': [(100, ), (300, ), (500, ), (800, )],
                    'activation' : ['relu', 'logistic'],#, 'tanh', 'identity'],
                    'solver' : ['sgd', 'adam', 'lbfgs'], 
                    'alpha' : [0.001, 0.005, 0.01, 0.1],
                    #'batch_size' : [400,600,800],
                    }
    elif (model == 'rf'):
      param_grid = {'n_estimators': [30, 50, 70, 100],
                    #'n_estimators': [10, 30, 50],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    "max_depth": [3, 5, 7],
                    #"min_samples_leaf": [1, 3, 5],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    }
    elif (model == 'svm'):
        param_grid = {'C' : [5, 7, 9, 12],
                    'kernel': ['rbf', 'poly'],#, 'sigmoid'],
                    'gamma': [5e-3, 1e-2, 1e-1],
                    'degree': [3, 5, 7, 9],
                    #'max_iter' : [100, 200, 300]
                    }
    return param_grid


def eval(clf, act, pred):
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(act, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(act, pred))


def scores(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return np.mean(scores)


def clear_txt_files(model_names):
    for model_name in model_names:
        file_name = "../classifiers/results/" + model_name + "-results.txt" 
        with open(file_name, "w") as f:
            f.write("RESULTS:\n\n")

    file_name = "../classifiers/results/best-results.txt" 
    with open(file_name, "w") as f:
        f.write("Best classifier results::\n\n")
    file_name = "../classifiers/results/summary.txt" 
    with open(file_name, "w") as f:
        f.write("Prediction Accuracy Summary:\n\n")


def write_results(act, pred, team_name, model_name):
    file_nmae = "../classifiers/results/" + model_name + "-results.txt" 
    with open(file_nmae, "a") as f:
      f.write(team_name + ":\n")
      f.write("----------------------------------------------------\n")
      f.write(metrics.classification_report(act,pred))
      f.write("\n\n")


def write_best_scores(scores, team_name, data):
    with open("../classifiers/results/best-results.txt", "a") as f:
        model_name, score, clf = scores[0]
        f.write(team_name + ":\n")
        f.write("----------------------------------------------------\n")
        f.write("model: %s\n" % model_name)
        f.write(str(clf))
        f.write("\n\n classes: %s" %(np.unique(data)))
        f.write("\n score: %f\n" %float(score))
        f.write("\n\n")
        file_name = "../classifiers/best/" + team_name + ".pkl"
        joblib.dump(clf, file_name)

def write_summary(scores, team_name, label_size, train_size, types):
    with open("../classifiers/results/summary.txt", "a") as f:
        table = "====================================================\n"\
                "  Pitcher: {0:12s}                                  \n\n"\
                "  Pitch Types:{6}                                   \n"\
                "  label_size:{4}   train_size:{5}                   \n"\
                "----------------------------------------------------\n"\
                "  SVM         NeuN       RanF                       \n"\
                "  =====       =====      =====                      \n"\
                "  {1:.3f}       {2:.3f}      {3:.3f}                \n"\
                "====================================================\n\n"\
                .format(team_name, float(scores[0][1]), float(scores[1][1]), float(scores[2][1]),\
                label_size, train_size, types)
        
        f.write(table)

if __name__ == "__main__":
    main()