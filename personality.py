import pandas as pd
import spacy
from spacy import displacy
NER = spacy.load("en_core_web_sm")

def get_personality_details(personality):
    detail_dict = {}
    personality = personality.upper()
    keywords = ['Personality_Profile', 'Strengths','Potential_development_areas', 'Typical_Characteristics', 'Career', 'Relationships', 'Stress_Behavior']

    df = pd.read_csv("data/Personality_details.csv")
    
    for keyword in keywords:
        # detail_dict[keyword] = ' '.join(df.query("`Personality Type`==@personality")[keyword].values)
        detail_dict[keyword] = df.query("`Personality Type`==@personality")[keyword].values[0]

    return detail_dict

def get_ner_html(user_input):
    doc = NER(user_input)
    html = displacy.render(doc, style="ent", jupyter=False)
    return html

