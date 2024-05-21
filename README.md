---

# Swasthya Bandhu - Health Assistant

Swasthya Bandhu is a Python-based diseases predictor tool and chatbot designed to provide users with information about various diseases, symptoms, and precautions. It leverages natural language processing (NLP) technique(NLTK) for chat interaction and machine learning for disease prediction.

## Features
 
- **Disease Prediction**: The chatbot can predict diseases based on the symptoms provided by the user, along with recommended precautions.
- **Chat Interface**: Users can interact with the chatbot through a web interface to get instant responses to their health-related queries.
- **Data Preprocessing**: The dataset is preprocessed to handle spelling mistakes and ensure accurate predictions.
- **Machine Learning Model**: Utilizes a Random Forest Classifier for disease prediction.
- **Web Framework**: Built using Flask, a lightweight Python web framework.

## Directory Structure

```
- templates/
  - index.html
  - chat_query.html
  - disease.html
- dataset.csv
- intents.json
- manipulated_dataset.csv
- swasthya_bandhu.py
- symptom_precaution.csv
```

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/sujitlayek21/swasthya-bandhu.git
   ```
2. Install dependencies:
   ```
   pip install flask scikit-learn pandas nltk
   ```
3. Run the `swasthya_bandhu.py` script to start the Flask server:
   ```
   python swasthya_bandhu.py
   ```
4. Access the chat interface by opening `http://localhost:5000/` in a web browser.


