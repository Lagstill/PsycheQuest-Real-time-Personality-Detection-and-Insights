import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
import pandas as pd
from joblib import load
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

mbti = [
    "INFP",
    "INFJ",
    "INTP",
    "INTJ",
    "ENTP",
    "enfp",
    "ISTP",
    "ISFP",
    "ENTJ",
    "ISTJ",
    "ENFJ",
    "ISFJ",
    "ESTP",
    "ESFP",
    "ESFJ",
    "ESTJ",
]

# part of speech dictionary
tags_dict = {
    "ADJ_avg": ["JJ", "JJR", "JJS"],
    "ADP_avg": ["EX", "TO"],
    "ADV_avg": ["RB", "RBR", "RBS", "WRB"],
    "CONJ_avg": ["CC", "IN"],
    "DET_avg": ["DT", "PDT", "WDT"],
    "NOUN_avg": ["NN", "NNS", "NNP", "NNPS"],
    "NUM_avg": ["CD"],
    "PRT_avg": ["RP"],
    "PRON_avg": ["PRP", "PRP$", "WP", "WP$"],
    "VERB_avg": ["MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    ".": ["#", "$", "''", "(", ")", ",", ".", ":"],
    "X": ["FW", "LS", "UH"],
}

# input to the model
features = [
      "clean_posts",
        "compound_sentiment",
        "ADJ_avg",
        "ADP_avg",
        "ADV_avg",
        "CONJ_avg",
        "DET_avg",
        "NOUN_avg",
        "NUM_avg",
        "PRT_avg",
        "PRON_avg",
        "VERB_avg",
        "em",
        "word_count",
        "unique_words"
]


def unique_words(s):
    unique = set(s.split(" "))
    return len(unique)


def emojis(post):
    # does not include emojis made purely from symbols, only :word:
    emoji_count = 0
    words = post.split()
    for e in words:
        if "http" not in e:
            if e.count(":") == 2:
                emoji_count += 1
    return emoji_count


def colons(post):
    # Includes colons used in emojis
    colon_count = 0
    words = post.split()
    for e in words:
        if "http" not in e:
            colon_count += e.count(":")
    return colon_count


def lemmitize(s):
    lemmatizer = WordNetLemmatizer()
    new_s = ""
    for word in s.split(" "):
        lemmatizer.lemmatize(word)
        if word not in stopwords.words("english"):
            new_s += word + " "
    return new_s[:-1]


def clean(s):
    # remove urls
    s = re.sub(re.compile(r"https?:\/\/(www)?.?([A-Za-z_0-9-]+).*"), "", s)
    # remove emails
    s = re.sub(re.compile(r"\S+@\S+"), "", s)
    # remove punctuation
    s = re.sub(re.compile(r"[^a-z\s]"), "", s)
    # Make everything lowercase
    s = s.lower()
    # remove all personality types
    for type_word in mbti:
        s = s.replace(type_word.lower(), "")
    return s


def prep_counts(s):
    clean_s = clean(s)
    d = {
        "clean_posts": lemmitize(clean_s),
        "link_count": s.count("http"),
        "youtube": s.count("youtube") + s.count("youtu.be"),
        "img_count": len(re.findall(r"(\.jpg)|(\.jpeg)|(\.gif)|(\.png)", s)),
        "upper": len([x for x in s.split() if x.isupper()]),
        "char_count": len(s),
        "word_count": clean_s.count(" ") + 1,
        "qm": s.count("?"),
        "em": s.count("!"),
        "colons": colons(s),
        "emojis": emojis(s),
        "unique_words": unique_words(clean_s),
        "ellipses": len(re.findall(r"\.\.\.\ ", s)),
    }
    return clean_s, d


def prep_sentiment(s):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(s)
    d = {
        "compound_sentiment": score["compound"],
        "pos_sentiment": score["pos"],
        "neg_sentiment": score["neg"],
        "neu_sentiment": score["neu"],
    }
    return d


def tag_pos(s):
    tagged_words = nltk.pos_tag(word_tokenize(s))
    d = dict.fromkeys(tags_dict, 0)
    for tup in tagged_words:
        tag = tup[1]
        for key, val in tags_dict.items():
            if tag in val:
                tag = key
        d[tag] += 1
    return d


def prep_data(s):
    clean_s, d = prep_counts(s)
    d.update(prep_sentiment(lemmitize(clean_s)))
    d.update(tag_pos(clean_s))
    return pd.DataFrame([d])[features]

def trace_back(combined):
    type_list = [
        {"0": "I", "1": "E"},
        {"0": "N", "1": "S"},
        {"0": "F", "1": "T"},
        {"0": "P", "1": "J"},
    ]
    result = []
    for num in combined:
        s = ""
        for i in range(len(num)):
            s += type_list[i][num[i]]
        result.append(s)
    return result

def combine_classes(y_pred1, y_pred2, y_pred3, y_pred4):
    combined = []
    for i in range(len(y_pred1)):
        combined.append(
            str(y_pred1[i]) + str(y_pred2[i]) + str(y_pred3[i]) + str(y_pred4[i])
        )
    result = trace_back(combined)
    return result[0]

def predict(s):

    X = prep_data(s)

    # loading the 4 models
    EorI_model = load("LR_model/clf_is_Extrovert.joblib")
    SorN_model = load("LR_model/clf_is_Sensing.joblib")
    TorF_model = load("LR_model/clf_is_Thinking.joblib")
    JorP_model = load("LR_model/clf_is_Judging.joblib")

    # predicting
    EorI_pred = EorI_model.predict(X)
    SorN_pred = SorN_model.predict(X)
    TorF_pred = TorF_model.predict(X)
    JorP_pred = JorP_model.predict(X)

    # print(EorI_pred, SorN_pred, TorF_pred, JorP_pred)
    # combining the predictions from the 4 models
    result = combine_classes(EorI_pred, SorN_pred, TorF_pred, JorP_pred)

    return result

def compute_personality_probabilities(data):
  mbti = [
    "INFP",
    "INFJ",
    "INTP",
    "INTJ",
    "ENTP",
    "ENFP",
    "ISTP",
    "ISFP",
    "ENTJ",
    "ISTJ",
    "ENFJ",
    "ISFJ",
    "ESTP",
    "ESFP",
    "ESFJ",
    "ESTJ",]
  res = {}
  for i in mbti:
    score = 0
    for j in list(i):
      score += data[j]
    res[i] = score
  return res  
        

def combine_classes_proba(data):    
  personality_dict = {}
  for i, subarray in enumerate(data):
      if subarray[0] > subarray[1]:
          if i == 0:
              personality_dict["E"] = subarray[0]
              personality_dict["I"] = subarray[1]
          elif i == 1:
              personality_dict["S"] = subarray[0]
              personality_dict["N"] = subarray[1]
          elif i == 2:
              personality_dict["F"] = subarray[0]
              personality_dict["T"] = subarray[1]
          elif i == 3:
              personality_dict["P"] = subarray[0]
              personality_dict["J"] = subarray[1]
      else:
          if i == 0:
              personality_dict["E"] = subarray[1]
              personality_dict["I"] = subarray[0]
          elif i == 1:
              personality_dict["S"] = subarray[1]
              personality_dict["N"] = subarray[0]
          elif i == 2:
              personality_dict["F"] = subarray[1]
              personality_dict["T"] = subarray[0]
          elif i == 3:
              personality_dict["P"] = subarray[1]
              personality_dict["J"] = subarray[0]
  return compute_personality_probabilities(personality_dict)


def predict_probabilty(s):

    X = prep_data(s)

    # loading the 4 models
    EorI_model = load("LR_model/clf_is_Extrovert.joblib")
    SorN_model = load("LR_model/clf_is_Sensing.joblib")
    TorF_model = load("LR_model/clf_is_Thinking.joblib")
    JorP_model = load("LR_model/clf_is_Judging.joblib")

    # predicting
    EorI_pred = EorI_model.predict_proba(X)
    SorN_pred = SorN_model.predict_proba(X)
    TorF_pred = TorF_model.predict_proba(X)
    JorP_pred = JorP_model.predict_proba(X)

    # print(EorI_pred, SorN_pred, TorF_pred, JorP_pred)
    # combining the predictions from the 4 models
    result = combine_classes_proba([EorI_pred.tolist()[0], SorN_pred.tolist()[0], TorF_pred.tolist()[0], JorP_pred.tolist()[0]])

    return result


def normalise_value(data):
    min_val = min(data.values())
    max_val = max(data.values())

    # Create an empty dictionary to store the normalized values
    normalized_data = {}

    # Loop through each key-value pair in the original dictionary
    for key, value in data.items():
        # Apply Min-Max normalization formula
        normalized_value = (value - min_val) / (max_val - min_val) * 99 + 1
        # Store the normalized value in the new dictionary
        normalized_data[key] = int(normalized_value)
    return normalized_data

def predict_reddit(comment):
    comment= " ".join(comment)
    # personality = predict(comment)
    res = predict_probabilty(comment)
    v = list(res.values())
 
    # taking list of car keys in v
    k = list(res.keys())
    
    personality = k[v.index(max(v))]
    print(res)
    print(normalise_value(res), personality)
    return normalise_value(res), personality
    

def predict_realtime(user_input):
    res = predict_probabilty(user_input)
    v = list(res.values())
    k = list(res.keys())
    personality = k[v.index(max(v))]
    return personality
