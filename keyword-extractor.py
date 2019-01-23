# coding: utf-8

import pandas as pd
import numpy as np

# load the dataset
papers_data = pd.read_csv('../../../Downloads/papers.csv')
# removing data which doesn't contain abstract
papers = papers_data.replace('Abstract Missing', "NaN")
dataset = papers.dropna()
dataset.index = np.arange(0, len(dataset))

""" Preliminary text exploration """
# Fetch word count for each abstract
dataset['word_count'] = dataset["abstract"].apply(lambda x: len(str(x).split(" ")))

# Descriptive statistics of word counts
dataset['word_count'].describe()

# Identify common words
common_words = pd.Series(''.join(dataset['abstract']).split()).value_counts()

# Identify uncommon words
uncommon_words = pd.Series(''.join(dataset['abstract']).split()).value_counts()[-20:]

""" Text Pre-processing """
# --- objective ----
# text clean-up
# shrinking the vocab to retaining only important world
# reduce sparsity
# --- task ----
# noise reduction
# normalization 1) stemming 2) lemmetization

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize

stem = PorterStemmer()
lem = WordNetLemmatizer()
# word = "dries"
# print("stemming:", stem.stem(word))
# print("lemmatisation:", lem.lemmatize(word, "v"))

# removing stopwords
stop_words = set(stopwords.words("english"))
# list of custom stop words: common words frequently occurs 1000 times
custom_words = list(common_words.index[common_words > 1000])
# combined stopwords
stop_words = stop_words.union(custom_words)

# cleaning and normalizing text corpus of data
corpus = []
dataset_length = dataset.__len__()
for i in range(0, dataset_length):
    # remove punctuation
    text = re.sub('[^a-zA-Z]', ' ', dataset['abstract'][i])

    # convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    ## convert to list from string
    text = text.split()

    ## stemming and lemmetization
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)
    corpus.append(text)

""" Data Exploration """
# word count
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

wordcloud = WordCloud(background_color="white", stopwords=stop_words, max_words=100,
                      max_font_size=50, random_state=42).generate(str(corpus))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("nips_word.png", dpi=900)

""" Text preparation """
# bag of word approach: considers word frequencies
from sklearn.feature_extraction.text import CountVectorizer
import re

cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(corpus)

# length of top 10 vocabulary
# list(cv.vocabulary_.keys())[:10]

""" Visualize top N uni-grams, bi-grams & tri-grams """
# Most frequently occurring words
def get_top_n_words(corpus, n=None, ngram=1):
    vec = CountVectorizer(ngram_range=(ngram, ngram), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

## uni-gram
# Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
# bar plot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize': (13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

## bi-gram
# Convert most freq words to dataframe for plotting bar plot
top2_words = get_top_n_words(corpus, n=20, ngram=2)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
# bar plot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize': (13,8)})
h = sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(g.get_xticklabels(), rotation=45)

## tri-gram
# Convert most freq words to dataframe for plotting bar plot
top3_words = get_top_n_words(corpus, n=20, ngram=3)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
# bar plot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize': (13,8)})
j = sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(g.get_xticklabels(), rotation=30)


""" converting matrix to integers """
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)
# get feature names
feature_names = cv.get_feature_names()
# fetch document for which keywords needs to be extracted
doc = corpus[75]
# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

# Function for sorting tf_idf in descending order
from scipy.sparse import coo_matrix


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())
# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 5)

# now print the results
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k, keywords[k])
