import praw, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime as dt
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image

reddit = praw.Reddit(client_id='pIFcQNQte4ajY8Bqzgqvug', \
                     client_secret='cwHurgF6CTwHPPXRZ3V2aDcoY2X6nw', \
                     user_agent='Alagu Prakalya', \
                     username='Efficiency_fresh', \
                     password='Testing@2002')


# praw.Reddit(client_id='w0cDom4nIf5druip4y9zSw', \
#                      client_secret='mtCul8hEucwNky7hLwgkewlLPzH0sg', \
#                      user_agent='Profile extractor', \
#                      username='CarelessSwordfish541', \
#                      password='Testing@2022')
analyzer = SentimentIntensityAnalyzer()

def get_icon(username):
    user = reddit.redditor(username)
    url = user.icon_img
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    #save the image
    img.save('static/images/user_icon.png')

def get_sentiment(username):
    user = reddit.redditor(username)
    sentiment = []
    for comment in user.comments.new(limit=100):
        sentiment.append(analyzer.polarity_scores(comment.body)['compound'])
    sentiment_df = pd.DataFrame(sentiment, columns=['sentiment'])
    sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')
    sentiment_df['count'] = 1
    sentiment_df = sentiment_df.groupby('sentiment').count().reset_index()
    res = []
    res.append(['Sentiment','Percent'])
    if len(sentiment_df['sentiment']) == 2:
        res.append([sentiment_df['sentiment'][0], sentiment_df['count'][0]])
        res.append([sentiment_df['sentiment'][1], sentiment_df['count'][1]])
    else:
        res.append(['negative',50])
        res.append(['positive',50])
    return res

def get_wordcloud(username):
    user = reddit.redditor(username)
    words = []
    for comment in user.comments.new(limit=100):
        words.append(comment.body)
    words = ' '.join(words)
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 'http', 'www', 'com', 'reddit', 'www.reddit', 'www.reddit.com', 'www.reddit.com/r', 'www.reddit.com/r/'])
    wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=800).generate(words)
    wordcloud.to_file('static/images/wordcloud.png')

def get_comments(username):
    user = reddit.redditor(username)
    comments = []
    for comment in user.comments.new(limit=100):
        comments.append(comment.body)
    return comments
        
def get_details(username):
    user = reddit.redditor(username)
    return (user.name, user.link_karma, user.comment_karma, user.trophies(), user.is_gold, user.is_mod)

   


