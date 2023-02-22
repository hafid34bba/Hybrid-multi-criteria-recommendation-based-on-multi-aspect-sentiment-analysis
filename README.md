# Hybrid-multi-criteria-recommendation-based-on-multi-aspect-sentiment-analysis

This repository talks about new approach of recommendation system based hybrid model Neural colaboratif filtering and Colaboratif filtering based Multi aspects sentiment analysis.

In this project, me and my teamate reached a new art of state.

<h1>Approach architecture</h1>

![](images/approach.png)


<h1>1 Sentimens analysis</h1> <br>

While our approach uses sentiment analysis scores in calculating similarity between users, we started our project by building sentiment analysis models.<br>
In order to have a good recommendation system, we must create a very accurated sentiment analysis model, and that's why we tried and fine tuned different approachs.<br>
<br>

<h2>1.2   Tfidf with fully connected layers. </h2>
TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information retrieval (IR) and machine learning, that can quantify the importance or relevance of string representations (words, phrases, lemmas, etc) in a document amongst a collection of documents.

The reason of choosing fully connected layer against LSTM is that tfidf does not give any sequential information.



