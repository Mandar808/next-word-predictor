import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mymodel.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Get max sequence length from model input shape
max_sequence_len = model.input_shape[1] + 1  # Adjusted for padding

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')

    predicted_probs = model.predict(sequence, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)  # Get word index with highest probability

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "Unknown"

# Streamlit UI
st.title("Next Word Prediction App")
st.write("Enter a phrase, and the model will predict the next word!")

# User input
input_text = st.text_input("Enter text:", "")

if st.button("Predict Next Word"):
    if input_text.strip():
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"Predicted Next Word: **{next_word}**")
    else:
        st.warning("Please enter some text.")

