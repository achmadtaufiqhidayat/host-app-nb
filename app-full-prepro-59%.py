from flask import Flask, render_template, url_for, request, flash
import tweepy
import re, string, csv, pickle, os
from os.path import join, dirname, realpath
# from pathlib import Path
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googletrans import Translator
from textblob import TextBlob
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os
import csv
from flask import flash
from googletrans import Translator
from googletrans.models import Translated

# from sklearn.metrics import precision_score, recall_score, f1_score
# from flask import Flask, render_template
# import matplotlib.pyplot as plt
# import string as str



#Preprocessing Twitter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

dataNormalisasi = pd.read_csv('D:\Skripsi\model\model-skr\kamus_singkatan.csv', sep=',')

# ubah data normalisasi menjadi dictionary
dictNormalisasi = dict(zip(dataNormalisasi['singkatan'], dataNormalisasi['asli']))

hasil_preprocessing = []

def normalisasi_kata(teks, dict_normalisasi):
    teks = teks.split()
    hasil = []
    for kata in teks:
        if kata.lower() in dict_normalisasi:
            hasil.append(dict_normalisasi[kata.lower()])
        else:
            hasil.append(kata)
    return " ".join(hasil)

from pathlib import Path
# cur_dir=Path.cwd()
#     #cwd = os.getcwd()
# csv_path=str(cur_dir)+"\\..\\static\\files\\Data Preprocessing.csv"



def preprocessing_twitter():
    # Membuat File CSV
    
    # d = pd.read_csv(csv_path)
    # writer = csv.writer(d)
    # file = open('E:\Archieve File\Analisis-Sentimen-Twitter-Flask\static/files/Data Preprocessing.csv', 'w', newline='', encoding='utf-8')
    # writer = csv.writer(file)
    csv_path = os.path.join(app.root_path, 'static', 'files', 'Data Preprocessing.csv')
    writer = csv.writer(open(csv_path, 'w', encoding='utf-8', newline=''))
    with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        hasil_labeling.clear()
        for row in readCSV:
            # proses cleansing
            # remove mention, link, hashtag
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row[2]).split())
            # remove number
            clean = re.sub("\d+", "", clean)
            # remove single char
            clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
            # remove multiple whitespace menjadi satu spasi
            clean = re.sub('\s+', ' ', clean)
            # remove punctuation (emoji)
            clean = clean.translate(clean.maketrans("", "", string.punctuation))

            # proses casefolding
            casefold = clean.casefold()

            # normalisasi kata
            normakisais = normalisasi_kata(casefold, dictNormalisasi)

            # proses tokenizing
            # tokenizing = nltk.tokenize.word_tokenize(casefold)
            tokenizing = nltk.tokenize.word_tokenize(normakisais)

            # proses stopword
            # mengambil data stop word dari library
            stop_factory = StopWordRemoverFactory().get_stop_words()
            # menambah stopword sendiri
            more_stop_word = ['&amp', 'ad', 'ada', 'ae', 'ah', 
                              'aja', 'ajar', 'ajar', 'amp', 'apa', 'aya', 'bab', 'bajo', 
                              'bar', 'bbrp', 'beda', 'begini', 'bgmn', 'bgt', 'bhw', 'biar', 'bikin', 
                              'bilang', 'bkh', 'bkn', 'bln', 'bnyk', 'brt', 'buah', 'cc', 'cc', 
                              'ckp', 'com', 'cuy', 'd', 'dab', 'dah', 'dan', 'dg', 'dgn', 'di', 
                              'dih', 'dlm', 'dm', 'dpo', 'dr', 'dr', 'dri', 'duga', 'duh', 'enth', 
                              'er', 'et', 'ga', 'gak', 'gal', 'gin', 'gitu', 'gk', 'gmn', 'gs', 
                              'gt', 'gue', 'gw', 'hah', 'hallo', 'halo', 'hehe', 'hello', 'hha', 
                              'hrs', 'https', 'ia', 'iii', 'in', 'ini', 'iw', 'jadi', 'jadi', 'jangn', 
                              'jd', 'jg', 'jgn', 'jls', 'kak', 'kali', 'kalo', 'kan', 'kch', 'ke', 
                              'kena', 'ket', 'kl', 'kll', 'klo', 'km', 'kmrn', 'knp', 'kok', 'kpd', 
                              'krn', 'kui', 'lagi', 'lah', 'lahh', 'lalu', 'lbh', 'lewat', 'loh', 'lu', 
                              'mah', 'mau', 'min', 'mlkukan', 'mls', 'mnw', 'mrk', 'n', 'nan', 'ni', 
                              'nih', 'no', 'nti', 'ntt', 'ny', 'nya', 'nyg', 'oleh', 'ono', 'ooooo', 
                              'op', 'org', 'pen', 'pk', 'pun', 'qq', 'rd', 'rt', 'sama', 'sbg', 'sdh', 
                              'sdrhn', 'segera', 'sgt', 'si', 'si', 'sih', 'sj', 'so', 'sy', 't', 'tak', 
                              'tak', 'tara', 'tau', 'td', 'tdk', 'tdk', 'thd', 'thd', 'thn', 'tindkn', 
                              'tkt', 'tp', 'tsb', 'ttg', 'ttp', 'tuh', 'tv', 'u', 'upa', 'utk', 'uyu', 
                              'viral', 'vm', 'wae', 'wah', 'wb', 'wes', 'wk', 'wkwk', 'wkwkwk', 'wn', 
                              'woiii', 'xxxx', 'ya', 'yaa', 'yah', 'ybs', 'ye', 'yg', 'ykm']
            # menggabungkan stopword library + milik sendiri
            data = stop_factory + more_stop_word

            dictionary = ArrayDictionary(data)
            str = StopWordRemover(dictionary)
            # stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))
            stop_wr = nltk.tokenize.word_tokenize(str.remove(normakisais))

            # proses stemming
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            # mamanggil fungsi stemming
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)

            tweets = [row[0], row[1], row[2], clean, casefold,normakisais, tokenizing, stop_wr, stemming]
            hasil_preprocessing.append(tweets)

            writer.writerow(tweets)
            flash('Preprocessing Berhasil', 'preprocessing_data')


hasil_labeling = []

def labeling_twitter():
    # Set the path for CSV files
    preprocessing_csv_path = Path("static/files/Data Preprocessing.csv")
    labeling_csv_path = Path(app.root_path) / 'static' / 'files' / 'Data Labeling.csv'

    # Initialize translator
    translator = Translator()

    try:
        # Open CSV files
        with preprocessing_csv_path.open('r', encoding='utf-8') as preprocessing_file, \
                labeling_csv_path.open('w', encoding='utf-8', newline='') as labeling_file:

            # Create CSV reader and writer
            csv_reader = csv.reader(preprocessing_file, delimiter=',')
            csv_writer = csv.writer(labeling_file)

            # Clear previous labeling results
            hasil_labeling.clear()

            # Iterate through rows in the preprocessing CSV
            for row in csv_reader:
                tweet = {}
                try:
                    # Translate the text to English
                    value = translator.translate(row[7], dest='en')
                except Exception as e:
                    print(f"Terjadi kesalahan {e}", flush=True)

                terjemahan = value.text
                data_label = TextBlob(terjemahan)

                # Determine sentiment
                if data_label.sentiment.polarity > 0.0:
                    tweet['sentiment'] = "Positif"
                elif data_label.sentiment.polarity == 0.0:
                    tweet['sentiment'] = "Netral"
                else:
                    tweet['sentiment'] = "Negatif"
# disini rownya diubah untuk keperluan pakar, dari 7 ke 2
                labeling = tweet['sentiment']
                tweets = [row[1], row[7], labeling]
                hasil_labeling.append(tweets)

                # Write the results to the labeling CSV
                csv_writer.writerow(tweets)

        flash('Labeling Berhasil', 'labeling_data')

    except Exception as e:
        print(f"Terjadi kesalahan {e}", flush=True)

#Klasifikasi

# Membuat variabel df
df = None
df2 = None

# menentukan akurasi 0
akurasi = 0

def proses_klasifikasi():
    global df
    global df2  
    global akurasi
    tweet = []
    y = []

    with open("static/files/Data Labeling.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            tweet.append(row[1])
            y.append(row[2])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(tweet)
    # tfidf = vectorizer.fit_transform(X_train)
    x = vectorizer.transform(tweet)


    # Inisialisasi TfidfVectorizer
    vectorizer = TfidfVectorizer()

# Fit dan transform data tweet
    x = vectorizer.fit_transform(tweet)

# Ambil matriks TF-IDF
    tfidf_matrix = x.toarray()

# Ambil kata-kata yang muncul di korpus
    terms =vectorizer.get_feature_names_out()
    

# Ambil DF (Document Frequency) untuk setiap kata
    df = vectorizer.idf_

# Membuat DataFrame dari hasil perhitungan
    data = {'Kata': terms, 'TF': tfidf_matrix.T.tolist(), 'DF': df, 'IDF': 1 + df}
    df_result = pd.DataFrame(data)

# Menampilkan DataFrame
    print (df_result)


    # Tampilkan hasil perhitungan TF-IDF untuk setiap kata
    # feature_names = vectorizer.get_feature_names_out()
    # idf_values = vectorizer.idf_
    # tfidf_result = dict(zip(feature_names, idf_values))
    # for kata, nilai in tfidf_result.items():
    #     print(f"Kata: {kata}, Nilai TF-IDF: {nilai}")
    # feature_names = vectorizer.get_feature_names_out()
    # idf_values = vectorizer.idf_
    # tfidf_result = pd.DataFrame({"Kata": feature_names, "Nilai TF-IDF": idf_values})
    # print(tfidf_result)
   

    # split data training dan testing 80:20
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # metode support vector machine kernel linear
    # clf = SVC(kernel="linear")
    # clf.fit(x_train, y_train)

    # predict = clf.predict(x_test)
    # report = classification_report(y_test, predict, output_dict=True)
# metode NB
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    
    predict = clf.predict(x_test)
    report = classification_report(y_test, predict, output_dict=True)

    # simpan ke csv
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv(
        'static/files/Data Klasifikasi.csv', index=True)

    pickle.dump(vectorizer, open('static/files/vec.pkl', 'wb'))
    pickle.dump(x, open('static/files/tfidf.pkl', 'wb'))
    pickle.dump(clf, open('static/files/model.pkl', 'wb'))

    #Confusion Matrix
    unique_label = np.unique([y_test, predict])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label),
        index=['pred:{:}'.format(x) for x in unique_label],
        columns=['true:{:}'.format(x) for x in unique_label]
    )   

    cmtx.to_csv(
        'static/files/Data Confusion Matrix.csv', index=True)

    df = pd.read_csv(
        'static/files/Data Confusion Matrix.csv', sep=",")
    df.rename(columns={'Unnamed: 0': ''}, inplace=True)

    df2 = pd.read_csv(
        'static/files/Data Klasifikasi.csv', sep=",")
    df2.rename(columns={'Unnamed: 0': ''}, inplace=True)

    akurasi = round(accuracy_score(y_test, predict) * 100, 2)

    kalimat = ""

    for i in tweet:
        s = ("".join(i))
        kalimat += s

    urllib.request.urlretrieve(
        "https://firebasestorage.googleapis.com/v0/b/sentimen-97d49.appspot.com/o/Circle-icon.png?alt=media&token=b9647ca7-dfdb-46cd-80a9-cfcaa45a1ee4", 'circle.png')
    mask = np.array(Image.open("circle.png"))
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200, background_color='white', mask=mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12, 10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.savefig('static/files/wordcloud.png')

    # diagram pie

    counter = dict((i, y.count(i)) for i in y)
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()

    positif = counter["Positif"] if isPositive == True else 0
    negatif = counter["Negatif"] if isNegative == True else 0
    netral = counter["Netral"] if isNeutral == True else 0

    sizes = [positif, netral, negatif]
    labels = ['Positif', 'Netral', 'Negatif']
    #add colors
    colors = ['#00FF00','#0000FF','#FF0000']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%',
            shadow=True,colors=colors, textprops={'fontsize': 20})
    plt.savefig('static/files/pie-diagram.png')

    # diagram batang
    # creating the bar plot

    plt.figure()

    plt.hist(y, color=('#0000FF'))

    plt.xlabel("Tweet tentang UU KES")
    plt.ylabel("Jumlah Tweet")
    plt.title("Presentase Sentimen Tweet")
    plt.savefig('static/files/bar-diagram.png')
    flash('Klasifikasi Berhasil', 'klasifikasi_data')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'farez'


# Upload folder
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/scraping', methods=['GET', 'POST'])
# def scraping():
#     if request.method == 'POST':
#         keyword = request.form.get('keyword')
#         jumlah = request.form.get('jumlah')
#         hasil_scraping.clear()
#         scraping_twitter_query(keyword, jumlah)
#         return render_template('scraping.html', value=hasil_scraping)
#     return render_template('scraping.html', value=hasil_scraping)

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            hasil_preprocessing.clear()
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if file and allowed_file(file.filename):
                file.filename = "Data Scraping.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                hasil_preprocessing.clear()
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('preprocessing.html')

        if request.form.get('preprocess') == 'Preprocessing Data':
            preprocessing_twitter()
            return render_template('preprocessing.html', value=hasil_preprocessing)

    return render_template('preprocessing.html', value=hasil_preprocessing)

@app.route('/labeling', methods=['GET', 'POST'])
def labeling():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            hasil_labeling.clear()
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if file and allowed_file(file.filename):
                file.filename = "Data Preprocessing.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                hasil_labeling.clear()
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('labeling.html')

        if request.form.get('labeling') == 'Labeling Data':
            labeling_twitter()
            return render_template('labeling.html', value=hasil_labeling)
            
    return render_template('labeling.html', value=hasil_labeling)




@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('klasifikasi.html')
            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('klasifikasi.html',)
            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('klasifikasi.html')
            if file and allowed_file(file.filename):
                file.filename = "Data Labeling.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('klasifikasi.html')

        if request.form.get('klasifikasi') == 'Klasifikasi Data':
            proses_klasifikasi()
            return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)
            
    if akurasi == 0:
        return render_template('klasifikasi.html')
    else:
        return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)

# @app.route('/klasifikasi', methods=['GET', 'POST'])
# def klasifikasi():
#     if request.method == 'POST':
#         if request.form.get('upload') == 'Upload Data':
#             file = request.files['file']
#             if not allowed_file(file.filename):
#                 flash('Format file tidak diperbolehkan', 'upload_gagal')
#                 return render_template('klasifikasi.html')
#             if 'file' not in request.files:
#                 flash('File tidak boleh kosong', 'upload_gagal')
#                 return render_template('klasifikasi.html',)
#             if file.filename == '':
#                 flash('File tidak boleh kosong', 'upload_gagal')
#                 return render_template('klasifikasi.html')
#             if file and allowed_file(file.filename):
#                 file.filename = "Data Labeling.csv"
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
#                 flash('File Berhasil di upload', 'upload_berhasil')
#                 df_result = proses_klasifikasi()
#                 return render_template('klasifikasi.html', df_result=df_result)

#         if request.form.get('klasifipykasi') == 'Klasifikasi Data':
#             proses_klasifikasi()
#             df_result = proses_klasifikasi()
#             return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values, df_result=df_result)
            
#     if akurasi == 0:
#         df_result = proses_klasifikasi()
#         return render_template('klasifikasi.html', df_result=df_result)
#     else:
#         df_result = proses_klasifikasi()
#         return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)@app.route('/visualisasi', df_result = df_result)
    
def visualisasi():
    return render_template('visualisasi.html')

@app.route('/tentang')
def modelpredict():
    return render_template('tentang.html')

if __name__ == "__main__":
    app.run(debug=True)
