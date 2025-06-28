import joblib
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# --------- Definisikan Fungsi Text Preprocessing ---------
def casefolding(text):
    return text.lower()

def text_normalize(text):
    return text  # implement sesuai kebutuhan (e.g. mengganti slang/typo)

def remove_stop_words(text):
    return text  # implement dengan daftar stopwords

def stemming(text):
    return text  # implement dengan stemmer seperti Sastrawi

def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

# --------- Load Model & Vocabulary ---------
model = joblib.load('model_1.joblib')
vocab = pickle.load(open('kbest_feature.pickle', 'rb'))

# --------- Streamlit Interface ---------
st.write('# Hotel Sentiment Analysis')

# Input dari pengguna
review_text = st.text_input('Masukan Review Terhadap Hotel Kami')

# Tombol prediksi
if st.button('Prediksi Sentimen'):
    if review_text.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        # Preprocessing teks
        preprocessed_text = text_preprocessing_process(review_text)

        # TF-IDF transform
        tfidf = TfidfVectorizer(vocabulary=set(vocab))
        transformed_input = tfidf.fit_transform([preprocessed_text])

        # Prediksi
        prediction = model.predict(transformed_input)

        # Interpretasi hasil
        sentiment = 'Positif' if prediction[0] == 0 else 'Negatif'
        

        st.subheader('Prediksi Sentimen:')
        st.success(f'{sentiment}')
