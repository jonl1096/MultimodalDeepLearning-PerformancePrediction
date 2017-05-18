from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import codecs
import re
import pdb
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    menu = 'Please select from the following options: \n'\
           '(1) parse tweets to a textual format\n'\
           '(2) transform tweets to LDA topic vectors\n'\
           '(3) transform articles to LDA topic vectors\n'\
           '(4) exit\n'

    while True:
        print(menu)
        option = raw_input("Enter option: ")
        if option == '1':
            parse_tweets()
        elif option == '2':
            with open("collect_twitter/data/orioles_tweets_processed.csv") as f:
                df = pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python')
                X = df['text']
                LDA(X)
        elif option == '3':
            with open("../../full_data/preprocessed_data/raw_articles.csv") as f:
                df = pd.read_csv(f, encoding='latin-1', sep=',', engine='python', index_col=False, header=None)
                X = df[1]
                X = X.fillna("None")
                LDA(X)
        elif option == '4':
            exit(1)
        else:
            print("invalid option")


def visualize(X_trans, n_topics):
    topic_names = [("Topic: %d"%i) for i in range(n_topics)]    
    # assign topics
    topic_labels = []
    for i in range(len(X_trans)):
        doc_topic = np.argmax(X_trans[i])
        topic_labels.append(np.argmax(X_trans[i]))
    #visualize_cor(topic_components, topic_names)
    #visualize_mat(results_df, column_names, topic_names)
    #visualize_pca(X_trans, topic_labels)
    #kmeans_clustering(X_trans)
    #gmm_clustering(X_trans)


def LDA(X):
    # params
    n_features = 1000
    n_topics = 50
    n_top_words = 20
    n_samples = len(X)

    # vectorize
    X_trans, topics, topic_components= fit_lda(X, n_features, n_topics, n_top_words, n_samples)
    
    # write transformed data
    #X_LDA = pd.concat([dates, pd.DataFrame(X_trans)], axis=1)
    #outfile_name = "full_data/articles_lda.csv"
    #X_LDA.to_csv(outfile_name, sep=",", index=False)
    
    # visualize(X_LDA, n_topics)


def parse_tweets():
    with open("collect_twitter/data/orioles_tweets_14.csv") as f:
        df = pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python')
    with open("collect_twitter/data/orioles_tweets_15.csv") as f:
        df = df.append(pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python'), ignore_index=True)
    with open("collect_twitter/data/orioles_tweets_16.csv") as f:
        df = df.append(pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python'), ignore_index=True)

    outputFile = codecs.open("collect_twitter/data/orioles_tweets_processed.csv", "w+", "utf-8")
    outputFile.write('date<;>text')
    dates = np.unique(df['date'])
    for date in dates:
        tweet = ""
        for index, row in df[df['date'] == date].iterrows():
            tweet += get_words(row['text']) + " "
        outputFile.write(('\n%s<;>%s<;>' % (date, tweet)))
    outputFile.close()

def get_words(doc):
    try:
        link_tag = 'http://'
        if link_tag in doc:
            doc = doc[:doc.index(link_tag)]
        return ' '.join(re.findall(r"(\w+)", doc)).lower()
    except:
        return ""

def kmeans_clustering(X):
    klist = []
    ilist = []
    for k in range(1,30):
        kmeans = KMeans(n_clusters=k, copy_x=False)
        kmeans.fit(X)
        klist.append(k)
        ilist.append(kmeans.inertia_)
        if True:
            C, L = kmeans.cluster_centers_, kmeans.labels_
    plt.figure();
    plt.plot(klist,ilist,'o-');
    plt.title("KMeans clustering inertia graph")
    plt.ylabel("Inertia")
    plt.xlabel("K (number of clusters)")
    plt.show()

def gmm_clustering(X):
    scores = []
    klist = []
    for k in range(1,30):
        gmm = mixture.GMM(n_components=k, covariance_type='full')
        gmm.fit(X)
        klist.append(k)
        scores.append(np.mean(gmm.score(X)))
    plt.figure();
    plt.plot(klist,scores,'o-');
    plt.title("GMM clustering log-likelihood graph")
    plt.ylabel("Log Likelihood")
    plt.xlabel("K (number of clusters)")
    plt.show()


def visualize_cor(topic_components, topic_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    corr_mat = np.corrcoef(topic_components)
    np.fill_diagonal(corr_mat, 0)
    plt.imshow(corr_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ax.set_yticks(np.arange(topic_components.shape[0]))
    ax.set_yticklabels(topic_names, rotation='horizontal', fontsize=10)
    ax.set_xticks(np.arange(topic_components.shape[0]))
    ax.set_xticklabels(topic_names, rotation=70, fontsize=8)
    plt.title("Correlation Matrix of Topics")
    plt.show()


def visualize_mat(results_df, column_names, topic_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(results_df, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ax.set_yticks(np.arange(results_df.shape[1]+1))
    ax.set_yticklabels(topic_names, rotation='horizontal', fontsize=11)
    ax.set_xticks(np.arange(len(column_names)))
    ax.set_xticklabels([(i.split('_')[1]) for i in column_names], rotation=70, fontsize=11)
    plt.title("Distribution of each user response")
    plt.show()

def visualize_pca(X_trans, topic_labels):
    # normalize X
    X = X_trans[:] - np.mean(X_trans[:])
    pca = decomposition.PCA(n_components=X[0,:].size)
    pca.fit(X)
    X_pca = pca.transform(X)
    E_vectors = pca.components_.T
    E_values = pca.explained_variance_
    print("Explained variance with 2 eigan vectors: %f%%" %np.sum(pca.explained_variance_ratio_[:2]))
    print("Explained variance with 3 eigan vectors: %f%%" %np.sum(pca.explained_variance_ratio_[:3]))

    plt.scatter(X_pca[:,0], X_pca[:,1], s=1, c=topic_labels, marker='o')
    plt.title('2 Principle Components Projection on Status Topic Distribution')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X_pca[:,1], X_pca[:,2], X_pca[:,3], s=1, c=topic_labels, marker='o')
    plt.title('3 Principle Components Projection on Status Topic Distribution')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    plt.show()


def fit_lda(X, n_features, n_topics, n_top_words, n_samples):
    print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
    # LDA
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                ngram_range=(1,1),
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(X)
    # fit
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    X = lda.fit_transform(tf)
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    topics = print_top_words(lda, tf_feature_names, n_top_words)
    return X, topics, lda.components_

def print_top_words(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        new_topic = " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(new_topic)
        topics.append(new_topic)
    print()
    return(topics)

if __name__ == '__main__':
    main()
