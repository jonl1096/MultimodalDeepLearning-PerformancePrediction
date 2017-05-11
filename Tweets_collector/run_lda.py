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
    parse_tweets()
    with open("data/orioles_tweets_processed.csv") as f:
        df = pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python')

    # get data
    dates = df['date']
    X = df['text']

    # params
    n_features = 1000
    n_topics = 50
    n_top_words = 20
    n_samples = len(X)

    # vectorize
    X_trans, topics, topic_components= fit_lda(X, n_features, n_topics, n_top_words, n_samples)

    # write transformed data
    outfile_name = "data/orioles_tweets_LDA.csv"    
    np.savetxt(outfile_name, X_trans, delimiter="<;>")
    
    # names
    '''
    topic_names = ["Family/Urgent", "Offence", "School Crime", 
                    "Police/Crime/Satire", "Donald Trump", "Teenage Abortion", 
                    "Marriage", "Crime witness", "Attack/Terrorism", "Hilary Clinton"]
    '''

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
    #plot_complex(X_trans, df, topic_names)

def parse_tweets():
    with open("data/orioles_tweets_14.csv") as f:
        df = pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python')
    with open("data/orioles_tweets_15.csv") as f:
        df = df.append(pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python'), ignore_index=True)
    with open("data/orioles_tweets_16.csv") as f:
        df = df.append(pd.read_csv(f, encoding='latin-1', sep='<;>', index_col=False, engine='python'), ignore_index=True)

    outputFile = codecs.open("data/orioles_tweets_processed.csv", "w+", "utf-8")
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

def get_total_sentiments(X_trans, df, sentiment):
    num_topics = len(X_trans[0])
    total_sentiments = [0] * num_topics
    for i in range(len(X_trans)):
        total_sentiments += X_trans[i] * df[sentiment][i]

    total_num = np.sum(total_sentiments) + 0.0
    total_num_prec = [0.0] * num_topics
    #print("*Total %s*" %sentiment)
    for i in range(len(total_sentiments)):
        #print("Topic %d: %.2f" %(i, total_sentiments[i] / total_num))
        #total_num_prec[i] = total_sentiments[i]
        #total_num_prec[i] = total_sentiments[i] / total_num
        total_num_prec[i] = np.log(total_sentiments[i])
    return total_num_prec

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            #ax.set_rgrids(range(1, 6), angle=angle, labels=variables)
            
        labels = [ [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] for i in range(10)]
        
        for ax, angle, label in zip(axes, angles, labels):
            ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 0.5)
            
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

def get_sentiments(X_trans, df, sentiment):
    num_topics = len(X_trans[0])
    total_sentiments = [0] * num_topics
    for i in range(len(X_trans)):
        total_sentiments += X_trans[i] * df[sentiment][i]

    total_num = np.sum(total_sentiments) + 0.0
    sentiment_for_all_topic = np.zeros(10)
    
    for i in range(len(total_sentiments)):
        sentiment_for_all_topic[i] = total_sentiments[i] / total_num
        
    return sentiment_for_all_topic

if __name__ == '__main__':
    main()
