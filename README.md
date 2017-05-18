# d2m-final-project

### Members:

|Name |  jhid | github |
| :---|  :--- |  :--- |
| Alex Ahn | sahn25 | [alexahn917](https://github.com/alexahn917) |
| Dylan Lewis | dlewis77 | [lefranchboi](https://github.com/lefranchboi) |
| Jonathan Liu | jliu118 | [jonl1096](https://github.com/jonl1096)  |
| Jered McInerney | dmciner1 | [dmcinerny](https://github.com/dmcinerney)  |

[Google Drive Folder](https://drive.google.com/open?id=0B4ieDXWtATqka0h1VUxPaVFuME0)

## Important Files:
* [Deep Autoencoder ipybn](https://github.com/jonl1096/d2m-final-project/blob/master/data_ETL/multimodal_deep_learning_transformation/autoencoder/testing/Deep%20Autoencoder.ipynb)


## Instructions:


### Data Extraction Pipelines:

* To collect Tweets from Twitter, run [run_twitter_14.sh](https://github.com/jonl1096/d2m-final-project/blob/master/data_ETL/collect_twitter/run_twitter_14.sh)
* To collect News Articles data from Bing, run [retrieve_articles.py](https://github.com/jonl1096/d2m-final-project/blob/master/data_ETL/collect_news_articles/retrieve_articles.py)
* To collect statistical data from MLB gameday websites, run [extract_pitches.R](https://github.com/jonl1096/d2m-final-project/blob/master/data_ETL/collect_mlb_statistics/extract_pitches.R)


### Latent Dirichilet Allication

* To perform LDA on variety of data, run [run_lda.py](https://github.com/jonl1096/d2m-final-project/blob/master/data_ETL/run_lda.py)


### Multimodal Deep Learning

* To encode bimodal/trimodal data with/without supervision for shared representation learning, run [multi_modal_rep_learning.py](https://github.com/jonl1096/d2m-final-project/blob/master/data_ETL/multimodal_deep_learning/multi_modal_rep_learning.py)

### Classifications:

* To train/test classifiers for each type of data representation using SVM, Random Forest, Multi Layered Percentron, run [classify.py](https://github.com/jonl1096/d2m-final-project/blob/master/Models/classify.py)

* To train/test Recurrent Neural Network for each type of data, run [rnn.py](https://github.com/jonl1096/d2m-final-project/blob/master/Models/rnn.py)