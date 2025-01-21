import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


word_index = imdb.get_word_index()
reversed_word_index = {value:key for key, value in word_index.items()}

model = load_model('imdb_review_sentiment_simple_rnn.h5')
model.get_weights()

max_len = 500

def decode_review(encode_review):
    return ' '.join([reversed_word_index.get(i-3, 2) for i in encode_review])


def preprocess_text(text):
    words = text.lower()
    encode_review = [word_index.get(word, 2) + 3 for word in words]

    padded_review = sequence.pad_sequences([encode_review], maxlen=500)

    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]


# Step 4: User Input and Prediction
# Example review for prediction
example_review = "This movie was fantastic! The acting was great and the plot was thrilling."

sentiment,score=predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {score}')


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')


