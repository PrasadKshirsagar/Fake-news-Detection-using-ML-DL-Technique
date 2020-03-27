from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn import model_selection, naive_bayes, svm
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt
import nltk
#import contractions
from keras.callbacks import TensorBoard
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Embedding
from keras.layers import Bidirectional
import re, string, unicodedata
from keras.layers import Input, Lambda
from sklearn.model_selection import train_test_split
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import LSTM, CuDNNLSTM, CuDNNGRU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Model


#Ignoring the warnings
import warnings
warnings.filterwarnings(action = 'ignore') 

#Tensorboard
#%load_ext tensorboard.notebook
#%tensorboard --logdir logs


##################### Training data ############################################################

#Reading the data
df2 = pd.read_csv('../input/big-dataset-btp/train_total.csv')

print(df2)
X = df2.iloc[:, 1:2].values
print(X)
X_train = []
for i in X:
    for j in i:
        X_train.append(str(j))
y_train = df2.iloc[:, 0].values
print(y_train)

def replace_contractions(sentence):
    """Replace contractions in string of text"""
    return contractions.fix(sentence)

  
def words_list(sample):
    words = []
    """Tokenising the corpus"""
    for i in sample: 
        temp = []
        for j in word_tokenize(i):
            temp.append(j.lower())
        temp = normalize(temp)
        words.append(temp)
    return words

  
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

  
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

  
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

  
def lemmatize_words(words):
    """Lemmatize the words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas

 
def normalize(words):
    """This is the main function which takes all other functions to pre-process the data given"""
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    lemmatize_words(words)
    return words



#Performing preprocessing on the Training Set
vocab_size = 30000
maxlen = 20
embed_dim = 100
latent_dim = 100
batch_size = 300
class_num = 2
word_list = words_list(X_train)
X_train = []
for list_of_words in word_list:
    sentence = ' '.join(x for x in list_of_words)
    X_train.append(sentence)

X_train, y_train = shuffle(X_train, y_train)
#Managing the y vector
y_train = to_categorical(y_train, num_classes = 2)
print(y_train[0])




#Converting the training set into integers to be fed to the embedding layer
t = Tokenizer()
t.fit_on_texts(X_train)
encoded_word_list = t.texts_to_sequences(X_train)
X_train = pad_sequences(encoded_word_list, maxlen=maxlen, padding='pre')
print(X_train)


########################## Model ####################################################

#Encoder Model
encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
embedding_layer = Embedding(input_dim = vocab_size, output_dim = embed_dim, input_length = maxlen, name='Encoder-Embedding-Layer')
x1 = embedding_layer(encoder_inputs)
x1 = Dropout(0.2)(x1)
encoder = CuDNNLSTM(latent_dim, name='Encoder-LSTM', return_state = True)
encoder_outputs, state_h, state_c = encoder(x1)
encoder_states = [state_h, state_c]
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)


#Decoder Model
#Repeat Vector
decoder_input = RepeatVector(maxlen)(seq2seq_encoder_out[0])
decoder_LSTM = CuDNNLSTM(latent_dim, return_sequences=True, name='Decoder-LSTM')
decoder_LSTM_output = decoder_LSTM(decoder_input, initial_state=seq2seq_encoder_out)
y1 = Dropout(0.2)(decoder_LSTM_output)
decoder_max_output = GlobalMaxPooling1D()(y1)
decoder_dense = Dense(class_num, activation='softmax', name='Decoder-Softmax')
decoder_outputs = decoder_dense(decoder_max_output)


#### Seq2Seq Model ####
seq2seq_Model = Model(encoder_inputs,decoder_outputs)
seq2seq_Model.summary()



##################### Validation data ############################################################


dataset3 = pd.read_csv('../input/big-dataset-btp/val_total.csv')
X_test_init1 = dataset3.iloc[:, 1:2].values
y_test1 = dataset3.iloc[:, 0].values
X_test1 = []
for i in X_test_init1:
    for j in i:
        X_test1.append(str(j))
word_list1 = words_list(X_test1)
X_test1 = []
for list_of_words in word_list1:
    sentence = ' '.join(x for x in list_of_words)
    X_test1.append(sentence)
encoded_word_list1 = t.texts_to_sequences(X_test1)
X_val = pad_sequences(encoded_word_list1, maxlen=maxlen, padding='pre')

#Managing the y vector
y_val = to_categorical(y_test1, num_classes = 2)


#################################### Fitting Model ###################################################


seq2seq_Model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# checkpoint
from keras.callbacks import ModelCheckpoint
filepath="best_seq2seq_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Model fitting
history = seq2seq_Model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128,callbacks=callbacks_list, verbose=1)





###################################### Testing Model ##########################################

#Testing
dataset2 = pd.read_csv('../input/big-dataset-btp/fakenewsnet_test.csv')
X_test_init = dataset2.iloc[:, 1:2].values
y_test = dataset2.iloc[:, 0].values
X_test = []
for i in X_test_init:
    for j in i:
        X_test.append(str(j))
word_list = words_list(X_test)
X_test = []
for list_of_words in word_list:
    sentence = ' '.join(x for x in list_of_words)
    X_test.append(sentence)
encoded_word_list = t.texts_to_sequences(X_test)
X_test = pad_sequences(encoded_word_list, maxlen=maxlen, padding='pre')


#Managing the y vector
y_test = to_categorical(y_test, num_classes = 2)
score, acc = seq2seq_Model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print("The accuracy of the model on the test set is: ", acc)


##################### Confusion Matrix ############################################################


#Confusion Matrix
y_pred = seq2seq_Model.predict(X_test)
y_pred = [ np.argmax(t) for t in y_pred ]
y_test_non_category = [ np.argmax(t) for t in y_test]
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test_non_category, y_pred)
print(conf_mat)

