from flask import Flask, render_template, request
from nltk.chat.util import Chat, reflections
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import re

app = Flask(__name__)

# Chatbot setup
with open('intents.json', 'r') as file:
    json_data = json.load(file)

pairs = []
for entry in json_data['intents']:
    patterns = entry['patterns']
    responses = entry['responses']
    for pattern in patterns:
        pairs.append((pattern, responses))

chatbot = Chat(pairs, reflections)

# Disease prediction setup
df1 = pd.read_csv('dataset.csv')

# Preprocess the symptoms to remove leading/trailing spaces and correct spelling mistakes
'''def preprocess_symptoms(symptom_str):
    symptoms = [TextBlob(symptom.strip()).correct().string for symptom in symptom_str.split(',')]
    return ', '.join(symptoms)'''
def preprocess_symptoms(symptom_str):
    return symptom_str.replace('_', ' ').strip().lower()




df1['Symptoms'] = df1['Symptoms'].apply(preprocess_symptoms)
#df1 = df1[['Disease', 'Symptoms']]
df1.to_csv('manipulated_dataset.csv', index=False)

df_precaution = pd.read_csv('symptom_precaution.csv')
precaution_mapping = {disease: list(precautions) for disease, *precautions in df_precaution.itertuples(index=False)}

df = pd.read_csv('manipulated_dataset.csv')
X = df['Symptoms']
y = df['Disease']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vectorized, y)

@app.route('/')
def index():
    return render_template('index.html')

# Routes for chat
@app.route('/chat')
def chat_index():
    return render_template('chat_query.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    if user_input.lower() == 'bye':
        return "Goodbye!"
    elif user_input.lower() == 'hi':
        return "Hello!"
    response = chatbot.respond(user_input)
    if response:
        return response
    else:
        return "Sorry, I didn't understand that."

# Routes for disease prediction
@app.route('/disease')
def disease_index():
    return render_template('disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['symptoms']
        user_input = preprocess_symptoms(user_input)  # Preprocess user input
        user_input_vectorized = vectorizer.transform([user_input])
        probabilities = model.predict_proba(user_input_vectorized)[0]
        
        top_indices = probabilities.argsort()[::-1]
        top_diseases = label_encoder.inverse_transform(top_indices)
        top_confidences = [probabilities[i] * 100 for i in top_indices]
        
        # Filter results based on the confidence threshold
        confidence_threshold = 50.0
        filtered_results = [(disease, confidence) for disease, confidence in zip(top_diseases, top_confidences) if confidence >= confidence_threshold]
        
        if filtered_results:
            top_diseases, top_confidences = zip(*filtered_results)
            top_precautions = [precaution_mapping[disease] for disease in top_diseases]
            return render_template('disease.html', 
                                   top_diseases=top_diseases, 
                                   top_confidences=top_confidences,
                                   top_precautions=top_precautions, 
                                   symptoms=user_input)
        else:
            return render_template('disease.html', 
                                   message="The model is not confident enough in its prediction. Please provide more detailed symptoms.", 
                                   symptoms=user_input)

if __name__ == '__main__':
    app.run(debug=True)
