# Automated Keyword extraction from Articles using nlp
    In research & news articles, keywords form an important component since they provide a concise representation of 
    the article’s content. Keywords also play a crucial role in locating the article from information retrieval systems,
    bibliographic databases and for search engine optimization. Keywords also help to categorize the article into the 
    relevant subject or discipline.

    Conventional approaches of extracting keywords involve manual assignment of keywords based on the article content 
    and the authors’ judgment. This involves a lot of time & effort and also may not be accurate in terms of selecting 
    the appropriate keywords. With the emergence of Natural Language Processing (NLP), keyword extraction has evolved 
    into being effective as well as efficient.
    
 In this repository, we applying NLP on a collection of articles (more on this below) to extract keywords.
    

## Data Source:
NIPS Papers: Neural Info. Processing System Confrence papers from 1987 to 2016
https://www.kaggle.com/benhamner/exploring-the-nips-papers/data

## Credit:
This work has been recreated from this [article](https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34) just for test perpous. All credit goes to the writer of the article for detailed explanation. Please refer it to for detailed info and if you like the work give thumps-up to main author.

In this work some of the setting has been changed for experimental purpous, like `custom-stopword` that included all the freq. used word occuured more than 1000 times. 
