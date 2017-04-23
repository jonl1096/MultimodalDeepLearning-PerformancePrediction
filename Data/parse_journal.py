from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse, io
import json
import re
import requests

link1 = 'http://www.baltimoresun.com/sports/orioles/blog/bal-orioles-on-deck-what-to-watch-friday-vs-yankees-20170406-story.html'
link2 = 'https://winnersandwhiners.com/games/mlb/4-03-2017/blue-jays-vs-orioles-prediction-and-preview/'
link3 = 'http://thebaltimorewire.com/2017/02/28/baltimore-orioles-2017-mlb-predictions/'
link4 = 'http://www.redreporter.com/2017/4/18/15343400/reds-vs-orioles-game-1-preview-predictions-thread'

dates = ['date_1', 'date_2']
links = [[link1, link2], [link3, link4]]

def main():
    #retrieve_articles(dates, links)
    #vectorize()
    load()

def retrieve_articles(dates, links):
    # Read articles and save to text file
    open("articles.txt", "w")
    for day in range(len(dates)):
        print("Processing articles for %s" %dates[day])
        for link in links[day]:
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'html.parser')
            content = soup.find_all("p")
            f = open("articles.txt", "a")
            text = ""
            for par in content:
                text += par.get_text().encode('utf-8') + " "
            text = text.replace('\n', ' ')
            f.write(text)
        f.write('\n')
        f.close()

def vectorize():
    # Convert accumulated articles to bigram, stopworded vectors
    with open("articles.txt", "r") as f:
        articles = f.read().split('\n')
        if len(articles[-1]) < 1:
            articles = articles[:-1]

        count_vect = CountVectorizer(ngram_range=(1,2), stop_words='english')
        X_train_counts = count_vect.fit_transform(articles)

        # Transform to TF_IDF
        tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)

        # Save processed data
        io.mmwrite("vectorized_articles.mtx", X_train_tf)

def load():
    vect_articles = io.mmread("vectorized_articles.mtx")
    print(vect_articles)

if __name__ == '__main__':
    main()