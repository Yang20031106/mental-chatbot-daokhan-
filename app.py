import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# 1. Load models & data
# ----------------------------
intent_pipeline = pickle.load(open("model/intent_pipeline.pkl", "rb"))
emotion_pipeline = pickle.load(open("model/emotion_pipeline.pkl", "rb"))
faq_vectorizer = pickle.load(open("model/faq_vectorizer.pkl", "rb"))
df_faq = pd.read_pickle("model/faq_dataset.pkl")

# ----------------------------
# 2. Preprocessing function
# ----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    corrected = []
    for word in tokens:
        if word in stop_words or word.strip() == "":
            corrected.append(word)
        else:
            # pyspellchecker might return None, so we fallback to the original word
            c = spell.correction(word)
            corrected.append(c if c is not None else word)
    
    lemmatized = [lemmatizer.lemmatize(word) for word in corrected if word not in stop_words]
    return ' '.join(lemmatized)

# ----------------------------
# 3. Emotion templates
# ----------------------------
emotion_templates = {
    "anxious": "Take a deep breath and calm down. ",
    "depressed": "I understand you feel down. ",
    "happy": "Glad to hear from you! ",
    "angry": "I hear your frustration. ",
    "suicidal": "Please reach out to a professional immediately: "
}

# ----------------------------
# 4. Response function
# ----------------------------
def get_response(user_input):
    clean_input = preprocess_text(user_input)

    # Predict intent
    proba = intent_pipeline.predict_proba([clean_input])[0]
    best_idx = proba.argmax()
    intent_conf = proba[best_idx]
    intent = intent_pipeline.classes_[best_idx]
    
    if intent_conf < 0.4:
        intent = "fallback"

    # Predict emotion
    proba_e = emotion_pipeline.predict_proba([clean_input])[0]
    best_idx_e = proba_e.argmax()
    emotion_conf = proba_e[best_idx_e]
    emotion = emotion_pipeline.classes_[best_idx_e]
    
    if emotion_conf < 0.2:
        emotion = "neutral"

    # Generate response
    if intent == "faq":
        user_vector = faq_vectorizer.transform([clean_input])
        similarities = cosine_similarity(user_vector, faq_vectorizer.transform(df_faq['clean_question']))
        best_idx = similarities.argmax()
        answer = df_faq.iloc[best_idx]['answer']
        response = f"{emotion_templates.get(emotion,'')}{answer}"
    elif intent == "greeting":
        response = f"{emotion_templates.get(emotion,'')}Hello! How can I help you today?"
    elif intent == "farewell":
        response = f"{emotion_templates.get(emotion,'')}Goodbye! Take care."
    elif intent == "fallback":
        response = "I'm not sure I understand. Can you rephrase your question?"
    else:
        response = f"{emotion_templates.get(emotion,'')}I'm here to listen. Tell me more."

    return response, intent, intent_conf, emotion, emotion_conf

# ----------------------------
# 5. Streamlit UI
# ----------------------------
st.title("🧠 Mental Health Chatbot")
st.write("Talk to the bot about your mental health, stress, or questions.")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input.strip() != "":
        reply, intent, intent_conf, emotion, emotion_conf = get_response(user_input)
        st.text_area("Bot:", value=reply, height=150, disabled=True)
        st.markdown(f"**Intent:** {intent} (score: {intent_conf:.2f})")
        st.markdown(f"**Emotion:** {emotion} (score: {emotion_conf:.2f})")
    else:
        st.warning("Please type something to chat.")









