# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:35:08 2020

@author: Srijha Kalyan
"""
#app.py is backend while home.html, predict.html are frontend

from flask import Flask,render_template,request # we can call a web page using render template to initiaize flask class
import itertools
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
#import argparse
import regex as re
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
wordnet_lemmatizer = WordNetLemmatizer()

import pandas as pd
app = Flask(__name__)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

windows_size= 10
vocab_size_stop= 1579
embedding_size_glove=100
vocab_size= 1484

#embedding_matrix_lp=np.load('embedding_matr.npy')
def text_to_wordlist(text, remove_stopwords=True, stem_words=False):    
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = stopwords.words("english")
        text = [wordnet_lemmatizer.lemmatize(w) for w in text if not w in stops ]
        text = [w for w in text if w != "nan" ]
    else:
        text = [wordnet_lemmatizer.lemmatize(w) for w in text]
        text = [w for w in text if w != "nan" ]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    
    text = re.sub(r"\<", " ", text)
    text = re.sub(r"\>", " ", text)
    
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def app_test(sentence,h5file_paths):
    sentence_stop_removed= text_to_wordlist(sentence)
    answer= tokenizer.texts_to_sequences(sentence_stop_removed)
    answer = list(itertools.chain(*answer))
    answer =  pad_sequences([answer], value=0, padding="post", maxlen=windows_size).tolist()
    answer = np.asarray(answer)
    
    model1= load_model(h5file_paths[0])
    model2= load_model(h5file_paths[1])
    model3= load_model(h5file_paths[2])
    model4= load_model(h5file_paths[3])
    model5= load_model(h5file_paths[4])
    model6= load_model(h5file_paths[5])
    model7= load_model(h5file_paths[6])
    

    model_pred1= np.argmax(model1.predict(answer))
    model_pred2= np.argmax(model2.predict(answer))
    model_pred3= np.argmax(model3.predict(answer))
    model_pred4= np.argmax(model4.predict(answer))
    model_pred5= np.argmax(model5.predict(answer))
    model_pred6= np.argmax(model6.predict(answer))
    model_pred7= np.argmax(model7.predict(answer))
    
    #model_prediction = model_pred1
    model_prediction= np.round_( (model_pred1+model_pred2+model_pred3+model_pred4+model_pred5+model_pred6+model_pred7)/(len(h5file_paths)))
    return model_prediction

from flask_ngrok import run_with_ngrok
from flask import Flask
app = Flask(__name__,static_url_path='/static')
run_with_ngrok(app)   #starts ngrok when the app is run
@app.route('/')
def home():
    return render_template("index.html") 
    #return "<h1>Running Flask!</h1>"

@app.route('/predict', methods=['GET','POST'])  #decorator 

def predict():
    print("i was here 1")
    model_prediction= 2

    #model_prediction= random.choice([1,2,4])
    if request.method == "POST":
        
        #print(gru_load_model.input,gru_load_model.output)
        sentence = request.form.get("sentence")
        
        #file_list = ['model_glove_lstm.h5']
        file_list=['model_2lstm_b.h5','model_bilstm_a_b.h5','model_cnn.h5','model_glove_lstm.h5','model_glove_lstm_b.h5','model_gru.h5','model_lstm_cnn.h5']
        model_prediction= app_test(sentence,file_list)
        

    if (model_prediction==0):
        return render_template('index.html', prediction = 'Level {} - No signs of depression is detected'.format(model_prediction))
    if (model_prediction==1):
        return render_template('index.html', prediction = 'Level {} - Mild symptoms of depression is detected'.format(model_prediction))
    if (model_prediction==2):
        return render_template('index.html', prediction = 'Level {} - Moderate symptoms of depression is detected'.format(model_prediction))
    if (model_prediction==3):
        return render_template('index.html', prediction = 'Level {} - Moderately severe signs of depression is detected.\n You need to get yourself checked'.format(model_prediction))
    if (model_prediction==4):
        return render_template('index.html', prediction = 'Level {} - Severe symptoms of depression is detected.\n You need to go have a check'.format(model_prediction))
if __name__ == "__main__":
    app.run()





