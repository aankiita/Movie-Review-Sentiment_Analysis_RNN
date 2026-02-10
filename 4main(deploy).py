import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
import streamlit as st

# ----------------------------------------------------------------------------------
# FIX: Define a custom class to handle the version mismatch
# This class catches the deprecated 'time_major' argument and removes it.
class CompatibleSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        # Remove the 'time_major' argument if it exists, as Keras 3 doesn't support it
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)
# ----------------------------------------------------------------------------------

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model using the custom object
# We map the saved 'SimpleRNN' layer to our new 'CompatibleSimpleRNN' class
try:
    model = load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': CompatibleSimpleRNN},compile=False)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.9 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write('Please enter a movie review.')