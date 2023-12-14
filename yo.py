import streamlit as st
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def load_slang_mapping(file_path):
    slang_mapping = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                key, value = line.split(maxsplit=1)
                slang_mapping[key] = value
            except ValueError:
                print(f"Warning: Invalid format on line {line_number}")
    return slang_mapping
slang_mapping = load_slang_mapping('kbba.txt')

def remove_hash_tag_user_tag(text):
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text

def correct_slang_words(text, slang_mapping):
    corrected_words = [slang_mapping.get(word, word) for word in text]
    return corrected_words

def text_normalization(text, slang_mapping):
    text = remove_hash_tag_user_tag(text.lower())  # remove hashtag and usertag
    text = re.sub(r'http\S+', '', text)  # remove link
    # remove emoji
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+', '', text)
    # tokenization
    morph = nltk.word_tokenize(text)
    morph = correct_slang_words(morph, slang_mapping)  # correct slang word
    # stopword n punctuations
    stop = set(stopwords.words('indonesian') + list(string.punctuation))
    # steeming
    Fact = StemmerFactory()
    Stemmer = Fact.create_stemmer()
    words = [Stemmer.stem(x) for x in morph if x not in stop]
    # delete word included num
    words = [word for word in words if not re.match(r'.*\d.*', word)]
    return " ".join(words)

def preprocess_data(df, slang_mapping):
    result = []
    for text in df['full_text']:
        clear_text = text_normalization(text, slang_mapping)
        result.append(clear_text)
    df['cleaned'] = result
    return df

def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        loaded_model, loaded_vectorizer = joblib.load(model_file)
    return loaded_model, loaded_vectorizer

def label_data(text, model, vectorizer):
    # Transformasi teks menggunakan vectorizer yang telah di-fit
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return prediction[0]

def main():
    st.title("Aplikasi Streamlit untuk Input CSV dengan Preprocessing dan Pelabelan Otomatis")

    # Mendapatkan file CSV dari user
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file, encoding='latin1')

        # Menampilkan data DataFrame
        st.write("Data yang diimpor:")
        st.write(df)

        # Menghapus duplikat berdasarkan kolom 'full_text'
        df_no_duplicates = df.drop_duplicates(subset='full_text').copy()

        # Preprocessing data
        st.write("Data setelah preprocessing:")
        df_preprocessed = preprocess_data(df_no_duplicates, slang_mapping)
        st.write(df_preprocessed)

        # Memuat model menggunakan joblib
        model_path = 'lstm_model.joblib'  # Ganti dengan path dan nama file model Anda
        loaded_model, loaded_vectorizer = load_model(model_path)

        # Pelabelan otomatis
        df_preprocessed['predicted_label'] = df_preprocessed['cleaned'].apply(lambda x: label_data(x, loaded_model, loaded_vectorizer))

        # Menyatukan data awal dan kolom predicted_label
        df_result = pd.concat([df, df_preprocessed['predicted_label']], axis=1)

        # Menampilkan hasil
        st.write("Data Awal dengan Label yang Sudah Diprediksi:")
        st.write(df_result[['full_text', 'predicted_label']])

        # Visualisasi pie chart
        st.write("Visualisasi Hasil Prediksi Label:")
        labels_count = df_result['predicted_label'].value_counts()

        fig, ax = plt.subplots()
        ax.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3, edgecolor='w'))
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_facecolor('none')  # Set background color to transparent
        st.pyplot(fig)

        # Save labeled data to CSV
        df_preprocessed.to_csv('labeled_data.csv', index=False)

if __name__ == "__main__":
    main()
