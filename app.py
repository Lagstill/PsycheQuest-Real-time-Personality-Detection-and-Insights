from flask import Flask, render_template, request, redirect, url_for, flash, session
import requests, random, torch
from transformers import AutoTokenizer
from flask import jsonify
from reddit_mining import get_icon, get_sentiment, get_wordcloud, get_comments, get_details
from predictions import predict_reddit,predict_realtime
from personality import get_personality_details,get_ner_html
from dialogpt import generate_dialogue

app = Flask(__name__, template_folder='./templates',static_folder='./static')
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    
@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/Info', methods = ['GET','POST'])
def Info():
    return render_template('Info.html')

@app.route('/RedditAnalyser', methods = ['GET','POST'])
def RedditAnalyser():
    userDetails = {}
    username = "Warlizard"
    userDetails['userid'] = username
    icon = get_icon(username)
    detail = get_details(username)
    sentiment = get_sentiment(username)
    userDetails['sentiment'] = sentiment
    userDetails['detail'] = detail
    wordcloud = get_wordcloud(username)
    comments = get_comments(username)
    userDetails['probability'],userDetails['personality'] = predict_reddit(comments)
    userDetails['comments'] = comments
    userDetails['personality_details'] = get_personality_details(userDetails['personality'])
    if request.method == 'POST':
        userDetails = {}
        userID = request.form
        userDetails['userid'] = userID['userid']
        icon = get_icon(userID['userid'])
        detail = get_details(userID['userid'])
        sentiment = get_sentiment(userDetails['userid'])
        userDetails['sentiment'] = sentiment
        wordcloud = get_wordcloud(userDetails['userid'])
        userDetails['detail'] = detail
        comments = get_comments(userDetails['userid'])
        userDetails['comments'] = comments
        userDetails['probability'],userDetails['personality'] = predict_reddit(comments)
        userDetails['personality_details'] = get_personality_details(userDetails['personality'])
        return render_template('Reddit_Analyser.html', userDetails = userDetails)
    return render_template('Reddit_Analyser.html', userDetails = userDetails)

@app.route('/RealtimePrediction', methods = ['GET','POST'])
def RealtimePrediction():
    userDetails = {}
    text = "I am Alagu. I don't like to roam around. I have not been to Japan and India. I dont enjoy meeting people and I wish to see President Obama in alive press conference :)"
    userDetails['user-input'] = text
    userDetails['Personality'] = 'INTJ' #predict_realtime(text)
    userDetails['Personality_details'] = get_personality_details(userDetails['Personality'])
    userDetails['ner_html'] = get_ner_html(text)
    if request.method == 'POST':
        userDetails = {}
        user_input = request.form['user-input']
        userDetails['Personality'] = predict_realtime(user_input)
        userDetails['Personality_details'] = get_personality_details(userDetails['Personality'])
        userDetails['ner_html'] = get_ner_html(user_input)
        # print("***************",userDetails['Personality'])
        return render_template('Realtime_Prediction.html', userDetails = userDetails)
    return render_template('Realtime_Prediction.html', userDetails = userDetails)

@app.route('/ChatBot', methods = ['GET','POST'])
def ChatBot():
    # return render_template('Chat_Bot.html')
    if request.method == 'GET':
        return render_template('Chat_Bot.html')
    elif request.method == 'POST':
        data = request.json
        message = data['message']
        personality = data['personality']
        chat_history_ids = torch.tensor(tokenizer.encode("Hi, how can I help you today?", return_tensors='pt'))
        response_text = generate_dialogue(personality,message,chat_history_ids, tokenizer)
    return jsonify({'message': response_text, 'personality': personality})


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    # Set the session cookie to be secure
    app.config['SESSION_TYPE'] = 'filesystem'
    # Set the debug flag to true
    app.debug = True    
    # Run the app :)
    app.run()