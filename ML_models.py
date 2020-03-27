import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk


def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)



Corpus = pd.read_csv(r"../input/big-dataset-btp/train_val_total.csv",encoding='latin-1')

# Step - a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
#Corpus['text'].apply(lambda entry: entry.lower())
#Corpus['text'] = Corpus['text'].str.lower()
Corpus['text'] = Corpus.text.apply(custom_tokenize) 

#Corpus['text']= str([word_tokenize(entry) for entry in Corpus['text']])
#
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
#Corpus['text'].apply(lambda entry: word_tokenize(entry))
#Corpus['ttext'] = Corpus['text'].apply(lambda row: nltk.word_tokenize(row['text']))
#
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
#Corpus['text'] = Corpus['text'].astype(object)
#for index,entry in enumerate(Corpus['text']):
#    temp = nltk.word_tokenize(Corpus["text"][index])
#    Corpus.at[index,'text'] = temp

for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    print(entry)
    if type(entry) == float and np.isnan(entry):
        Corpus.loc[index,'text_final'] = str([''])
        continue
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word in entry:
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word)
            Final_words.append(word_Final)
            #print(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)





    

Corpus1 = pd.read_csv(r"../input/big-dataset-btp/liar_test.csv",encoding='latin-1')
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently

# Step - a : Remove blank rows if any.
Corpus1['text'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
#Corpus['text'].apply(lambda entry: entry.lower())
#Corpus['text'] = Corpus['text'].str.lower()
Corpus1['text'] = Corpus1.text.apply(custom_tokenize) 
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
for index,entry in enumerate(Corpus1['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    print(entry)
    if type(entry) == float and np.isnan(entry):
        Corpus1.loc[index,'text_final'] = str([''])
        continue
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word in entry:
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word)
            Final_words.append(word_Final)
            #print(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus1.loc[index,'text_final'] = str(Final_words)





#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.001)
#Train_X1, Test_X, Train_Y1, Test_Y = model_selection.train_test_split(Corpus1['text_final'],Corpus1['label'],test_size=0.99)



Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)




Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)



###################################### Naive Bayes ##############################################

clf = naive_bayes.MultinomialNB()
clf.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = clf.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)



###################################### Random Forest ##############################################
clf=RandomForestClassifier(n_estimators=20)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(Train_X_Tfidf,Train_Y)
y_pred=clf.predict(Test_X_Tfidf)
print("RF Accuracy Score -> ",accuracy_score(y_pred, Test_Y)*100)



###################################### Decision Tree ##############################################

clf = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 5000) 
  
# Performing training 
clf.fit(Train_X_Tfidf,Train_Y)     
y_pred=clf.predict(Test_X_Tfidf)
print("DT Accuracy Score -> ",accuracy_score(y_pred, Test_Y)*100) 


###################################### SVM ########################################################

SVM = svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)




