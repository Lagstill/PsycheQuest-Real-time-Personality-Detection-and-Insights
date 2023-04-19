import pandas as pd
# from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# load the dataframe with the data
df = pd.read_csv('data/Personality_details.csv')
keywords = ['Personality_Profile', 'Strengths',
            'Potential_development_areas', 'Typical_Characteristics', 'Career',
            'Under_Stress', 'Relationships', 'Stress_Behavior']

# define a function to check if any of the keywords are mentioned in the user's input
def check_keywords(input_text):
    for keyword in keywords:
        if keyword.lower() in input_text.lower():
            return keyword
    return None

# define a function to fetch the data from the dataframe based on the keyword
def fetch_data(personality, keyword):
    personality = personality
    return df.query("`Personality Type`==@personality")[keyword].values[0]

# define a function to generate a response based on the user's input
def generate_response(personality,input_text, chat_history_ids, tokenizer):
    keyword = check_keywords(input_text)
    if keyword:
        data = fetch_data(personality, keyword)
        response = "Here is some information on " + keyword + ":\n"
        response += "".join(data)
        chat_history_ids = None  # reset chat history for new topic
    else:
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids


def generate_dialogue(personality,user_input, chat_history_ids, tokenizer):
    response, chat_history_ids = generate_response(personality,user_input, chat_history_ids, tokenizer)
    # print("USER INPUT:", user_input)
    # print("Chatbot:", response)
    return response
