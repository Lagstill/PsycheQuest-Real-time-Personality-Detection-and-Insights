# Personality-Detection
This is a Flask-based personality detection project with features including real-time prediction, Reddit mining chatbot, NER tag identification. Personality detection is important for understanding human behavior, improving communication and relationships, and even enhancing mental health treatment.

![](https://github.com/Lagstill/Personality-Detection/blob/main/images/home.png)


## Features
Real-time prediction: Get instant results for MBTI personality type based on user input.
Reddit mining and MBTI prediction: Get MBTI personality types of users based on their Reddit comments.
Chatbot: Ask anything and the chatbot will answer you based on pre-defined responses.
NER tag identification: Identify Named Entities in the input text and highlight them.
General information page on MBTI: Learn more about the Myers-Briggs Type Indicator (MBTI).
EDA notebook: Explore and preprocess MBTI data and train a logistic regression model with TF-IDF and BERT.

## Installation
Clone the repository and install the required packages using pip:
```
$ git clone https://github.com/your_username/personality-detection.git
$ cd personality-detection
```

## Usage
Start the Flask server by running the following command in your terminal:

```
$ python app.py
```
![](https://github.com/Lagstill/Personality-Detection/tree/main/images/realtime.png)

## Endpoints
The following endpoints are available:

* /: Home page with links to other endpoints.
* /RealtimePrediction: Real-time prediction page where users can input text and get their MBTI personality type.
* /RedditAnalyser: Reddit mining page where users can input a Reddit username and get their MBTI personality type based on their comments.
* /ChatBot: Chatbot page where users can ask questions and get pre-defined answers.
* /Info: MBTI information page with general information about the Myers-Briggs Type Indicator.


![](https://github.com/Lagstill/Personality-Detection/tree/main/images/reddit.png)

## Contributing
Contributions are welcome! If you have any suggestions or find a bug, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
