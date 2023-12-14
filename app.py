import streamlit as st
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import string
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')


def download_custom_stopwords(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        stopwords_text = response.text
        custom_stopwords = set(stopwords_text.splitlines())
        return custom_stopwords
    except requests.exceptions.RequestException as e:
        print("Gagal mengunduh daftar kata-kata stop words:", e)
        return set()


github_stopwords_url = 'daftar_stopword.txt'
custom_stopwords = download_custom_stopwords(github_stopwords_url)

stop_words = set(stopwords.words('indonesian'))
stop_words.update(custom_stopwords)

stemmer = PorterStemmer()


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
                print(
                    f"Warning: Invalid format on line {line_number}. Expected 2 values.")

    return slang_mapping


slang_mapping = load_slang_mapping('kbba.txt')


def correctSlangWords(text, slang_mapping):
    corrected_words = [slang_mapping.get(word, word) for word in text]
    return corrected_words


def clean_tweet(tweet):
    tweet = re.sub(r'@[\w]*', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.lower()
    tweet = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+', '', tweet)

    tokens = nltk.word_tokenize(tweet)
    tokens = correctSlangWords(tokens, slang_mapping)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    words = [stemmer.stem(x) for x in tokens]
    words = [word for word in words if not re.match(r'.*\d.*', word)]
    cleaned_tweet = ' '.join(words)

    return cleaned_tweet


# Assuming content is a list of tweets
content = ['your', 'list', 'of', 'tweets']
data_cleaned = pd.DataFrame(content, columns=['full_text'])
data_cleaned.dropna(inplace=True)

# Apply the clean_tweet function to the 'full_text' column
data_cleaned['cleaned_text'] = data_cleaned['full_text'].apply(clean_tweet)

# Display the cleaned data
print(data_cleaned)


def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        loaded_model, loaded_vectorizer = pickle.load(model_file)
    return loaded_model, loaded_vectorizer


def label_data(text, model, vectorizer):
    # Transformasi teks menggunakan vectorizer yang telah di-fit
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return prediction[0]


def main():
    st.title(
        "Aplikasi Streamlit untuk Input CSV dengan Preprocessing dan Pelabelan Otomatis")

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

        # Memuat model menggunakan pickle
        model_path = 'svm_model.pkl'  # Ganti dengan path dan nama file model Anda
        loaded_model, loaded_vectorizer = load_model(model_path)

        # Pelabelan otomatis
        df_preprocessed['predicted_label'] = df_preprocessed['cleaned'].apply(
            lambda x: label_data(x, loaded_model, loaded_vectorizer))

        # Menyatukan data awal dan kolom predicted_label
        df_result = pd.concat([df, df_preprocessed['predicted_label']], axis=1)

        # Menampilkan hasil
        st.write("Data Awal dengan Label yang Sudah Diprediksi:")
        st.write(df_result[['full_text', 'predicted_label']])

        # Visualisasi pie chart
        st.write("Visualisasi Hasil Prediksi Label:")
        labels_count = df_result['predicted_label'].value_counts()

        fig, ax = plt.subplots()
        ax.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%',
               startangle=90, wedgeprops=dict(width=0.3, edgecolor='w'))
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        st.pyplot(fig)

        # Save labeled data to CSV
        df_preprocessed.to_csv('labeled_data.csv', index=False)


if __name__ == "__main__":
    main()
