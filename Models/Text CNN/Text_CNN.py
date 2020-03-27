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



# Defining parameters
stops = set(stopwords.words('english'))
vocab_size = 20000
sequence_length = 200
embedding_dim = 100
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
train_data = pd.read_csv("../input/big-dataset-btp/train_total.csv")
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

num_filters=2
dropOut=0.50

# Creating our text-CNN model :
input = Input(shape=(sequence_length, ))
aft_emb = Embedding(vocab_size, embedding_dim)(input)
aft_emb = Dropout(dropOut)(aft_emb)

# Function to create convolution models
def models(kernal_size):
    x = Conv1D(nb_filter = num_filters, kernel_size = kernal_size, border_mode = 'valid', activation = 'relu',strides = 1)(aft_emb)
    x_out = GlobalMaxPooling1D()(x)
    return x_out

# Model1 of filters having kernel size as 2
x_out = models(2)

# Model2 of filters having kernel size as 3
y_out = models(3)

# Model3 of filters having kernel size as 4
z_out = models(4)

# Concatenate the outputs from above 3 models
concatenated = concatenate([x_out, y_out, z_out])

# Apply dense layers 
dense1 = Dense(250)(concatenated)
dense1 = LeakyReLU(alpha=0.05)(dense1)
out = Dense(2, activation = 'softmax', name = 'output_layer')(dense1)

# Get final model
merged_model = Model(input, out)

# Creating Tensorboard
#tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(merged_model.summary())


##################### Fitting Model ############################################################

# checkpoint
from keras.callbacks import ModelCheckpoint
filepath="best_CNN_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# Model fitting
history = merged_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128,callbacks=callbacks_list, verbose=0)



##################### Testing Model ############################################################

# Preprocessing Test data

# Checking on liar Dataset 

test_data = pd.read_csv("../input/big-dataset-btp/webis_test.csv") 

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



# Final evaluation of the model
scores = merged_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) 




