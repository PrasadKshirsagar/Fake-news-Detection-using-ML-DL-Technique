from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn import model_selection, naive_bayes, svm
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense,GlobalMaxPooling1D
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dropout,add
import numpy as np
from keras.layers import Input
import pandas as pd
from keras.layers import LeakyReLU
from nltk.corpus import stopwords
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Dropout, Embedding, Bidirectional,CuDNNLSTM
from keras.layers import Dense,LSTM




# Defining parameters
stops = set(stopwords.words('english'))
vocab_size = 10000
sequence_length = 25
embedding_dim = 50
num_of_classes = 2



# Preprocessing of the data
def preprocess(text):
    sequences = tokenizer.texts_to_sequences(text)
    X = pad_sequences(sequences, padding='pre', maxlen=sequence_length)
    X = np.array(X)
    return X

from nltk.corpus import stopwords
stop = stopwords.words('english')


##################### Training data ############################################################

# Get sentences and labels from file
train_data = pd.read_csv("../input/big-dataset-btp/train_total.csv", error_bad_lines=False)

train_data['text'].replace('', np.nan, inplace=True)
train_data.dropna(subset=['text'], inplace=True)

#train_data['text'] = train_data['text'].str.lower()

#train_data['text'] = train_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data = train_data["text"]
labels = np.array(train_data["label"])

texts = []
for itr in data:
    texts.append(" ".join([word for word in str(itr).split() if word not in stops]))

    
# Using tokenizer API for tokenization
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
#X_texts = preprocess(texts)
#y_labels = np_utils.to_categorical(labels , num_classes=num_of_classes)
X_train = preprocess(texts)
y_train = np_utils.to_categorical(labels , num_classes=num_of_classes)



# Split data into train & test sets
'''
X_train, X_test, y_train, y_test = train_test_split(X_texts, y_labels, stratify=y_labels,test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,test_size=0.125, random_state=42)
print(X_train.shape ,y_train.shape)
print(X_val.shape ,y_val.shape)

'''

##################### Validation data ############################################################


val_data = pd.read_csv("../input/big-dataset-btp/val_total.csv", error_bad_lines=False) 

val_data['text'].replace('', np.nan, inplace=True)
val_data.dropna(subset=['text'], inplace=True)

#val_data['text'] = val_data['text'].str.lower()

#val_data['text'] = val_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


val_x = val_data["text"]
v_labels = np.array(val_data["label"])

text = []
for sentence in val_x:
    text.append(" ".join([word for word in str(sentence).split() if word not in stops]))

X_val = preprocess(text)

y_val = np_utils.to_categorical(v_labels , num_classes=num_of_classes)


##################### Model ############################################################

"""Building the Model"""

dim = 100
dropout = 0.2
lstm_out = 400

def Build_Model(vocab_sizze, dim, dropout, lstm_out):
    """Defining the model"""
    #tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    model = Sequential()
    e = Embedding(input_dim = vocab_size, output_dim = dim, input_length = X_train.shape[1], dropout = dropout) 
    model.add(e)
    model.add(Bidirectional(CuDNNLSTM(lstm_out, return_sequences = True)))
    model.add(Dropout(dropout))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model

model = Build_Model(vocab_size, dim, dropout, lstm_out)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

##################### Fitting Model ############################################################



# checkpoint
from keras.callbacks import ModelCheckpoint
filepath="best_RNN_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Model fitting
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128,callbacks=callbacks_list, verbose=1)



##################### Testing Model ############################################################


test_data = pd.read_csv("../input/big-dataset-btp/webis_test.csv") 

print(test_data['text'][0])

test_data['text'].replace('', np.nan, inplace=True)
test_data.dropna(subset=['text'], inplace=True)

#test_data['text'] = test_data['text'].str.lower()

#test_data['text'] = test_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


test_x = test_data["text"]
labels = np.array(test_data["label"])

text = []
for sentence in test_x:
    text.append(" ".join([word for word in str(sentence).split() if word not in stops]))

X_test = preprocess(text)
#print(X_test)
y_test = np_utils.to_categorical(labels , num_classes=num_of_classes)

#X_test,t1 , y_test, t2 = train_test_split(X_test, y_test, stratify=y_test,test_size=0.80, random_state=42)




# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
