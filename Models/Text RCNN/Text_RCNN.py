from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn import model_selection, naive_bayes, svm
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
#import contractions
import tensorflow as tf
#from keras.callbacks import TensorBoard
#import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Bidirectional
import keras 
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import LSTM, CuDNNLSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import string
import pandas as pd
from keras import backend
from keras.layers import Conv1D, Dense, Input, Lambda, LSTM, CuDNNLSTM
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model


#Ignoring the warnings
import warnings
warnings.filterwarnings(action = 'ignore')




#Reading the data
df2 = pd.read_csv('../input/big-dataset-btp/train_total.csv')
X = df2.iloc[:, 1:2].values
X_train = []
for i in X:
    for j in i:
        X_train.append(str(j))
y_train = df2.iloc[:, 0].values

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
    words = to_lowercase(words)
    words = remove_stopwords(words)
    lemmatize_words(words)
    return words





vocab_size = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 20
embedding_dim = 100
batch_size = 128
word_list = words_list(X_train)
X_train = []
for list_of_words in word_list:
    sentence = ' '.join(x for x in list_of_words)
    X_train.append(sentence)
X_train, y_train = shuffle(X_train, y_train)



t = Tokenizer()
t.fit_on_texts(X_train)
encoded_word_list = t.texts_to_sequences(X_train)
X_train = pad_sequences(encoded_word_list, maxlen=maxlen, padding='pre')
print(X_train)
print(y_train)

#Managing the y vector
y_train = to_categorical(y_train, num_classes = 2)


############################### Model ###############################################

hidden_dim_1 = 10
hidden_dim_2 = 10
NUM_CLASSES = 2

document = Input(shape = (maxlen, ), dtype = "int32")
left_context = Input(shape = (maxlen, ), dtype = "int32")
right_context = Input(shape = (maxlen, ), dtype = "int32")
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
#mc = ModelCheckpoint('best_model_RCNN.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
embedder = Embedding(vocab_size, embedding_dim)
doc_embedding = embedder(document)
left_embedding = embedder(left_context)
right_embedding = embedder(right_context)
# I use LSTM RNNs instead of vanilla RNNs as described in the paper.
forward = CuDNNLSTM(hidden_dim_1, return_sequences = True)(left_embedding) # See equation (1).
backward = CuDNNLSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(right_embedding) # See equation (2).
# Keras returns the output sequences in reverse order.
backward = Lambda(lambda x: backend.reverse(x, axes = 1))(backward)
together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3).
dropout_1 = Dropout(0.2)(together)
semantic = Conv1D(hidden_dim_2, kernel_size = 3, activation = "relu")(dropout_1) # See equation (4).
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).
dropout_2 = Dropout(0.2)(pool_rnn)
output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(dropout_2) # See equations (6) and (7).

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()


########################################################################################


left_context_as_array = []
right_context_as_array = []
for i in X_train:
    j =  np.array(i).tolist()
    # We shift the document to the right to obtain the left-side contexts.
    left_context_as_array.append([0] + j[:-1])
    # We shift the document to the left to obtain the right-side contexts.
    right_context_as_array.append(list(j[1:] + [0]))
    

# checkpoint
from keras.callbacks import ModelCheckpoint
filepath="best_RCNN_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit([np.array(X_train), np.array(left_context_as_array), np.array(right_context_as_array)], y_train, epochs = 10, verbose = 1, batch_size = batch_size, validation_split=0.125,callbacks=callbacks_list)


###################################### Testing ##################################################

dataset2 = pd.read_csv('../input/big-dataset-btp/webis_test.csv')

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
X_test = pad_sequences(encoded_word_list, maxlen=20, padding='pre')
left_context_as_array_test = []
right_context_as_array_test = []
for i in X_test:
    j =  np.array(i).tolist()
    # We shift the document to the right to obtain the left-side contexts.
    left_context_as_array_test.append([0] + j[:-1])
    # We shift the document to the left to obtain the right-side contexts.
    right_context_as_array_test.append(list(j[1:] + [0]))
#Managing the y vector
y_test = to_categorical(y_test, num_classes = 2)
score, acc = model.evaluate([np.array(X_test), np.array(left_context_as_array_test), np.array(right_context_as_array_test)], y_test, batch_size = 64, verbose = 1)
print("The accuracy of the model on the test set is: ", acc)