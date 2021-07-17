import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import spacy 
import re

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, GlobalMaxPooling1D, BatchNormalization, Dropout, Flatten, Conv1D, MaxPooling1D

# Loading the English core large model trained on web articles
nlp = spacy.load('en_core_web_lg')


df = pd.read_csv('G:/Courses/Datasets/Twint/combined.csv')

texts = list(df['tweets'])
labels = list(df['depression'])

# Preprocessing the tweets, removing hyperlinks, keeping only alphanumeric characters, and removing usernames to maintain anonymity
def preprocess_texts(texts):
    cleaned_texts = []
    for i in range(len(texts)):
        temp = texts[i]
        temp = temp.lower()
        temp = re.sub('@[\w]+', ' ', temp)
        temp = re.sub('https?:\/\/\S+', ' ', temp)
        temp = re.sub("www\.[a-z]?\.?(com)+|[a-z]+\.(com)", ' ', temp)
        temp = re.sub('[^a-z]', ' ', temp)
        temp = re.sub(' +', ' ', temp)
        cleaned_texts.append(temp)
    return cleaned_texts 

cleaned_texts = preprocess_texts(texts)


# Some of the tweets given out by the preprocess_text function are empty, so we remove them 

to_be_removed = []
for i in range(len(cleaned_texts)):
    if cleaned_texts[i] == ' ':
        to_be_removed.append(i)     
 
final_texts = []
final_labels = []
for i in range(len(labels)):
    if i in to_be_removed:
        continue
    else:
        final_texts.append(cleaned_texts[i])
        final_labels.append(labels[i])



# Getting sentence vectors using the respective spaCy model 
def get_sentence_vectors(model, texts):
    sentence_vectors = []
    for i in range(len(texts)):
        doc = nlp(texts[i])
        sentence_vectors.append(doc.vector)
    
    return sentence_vectors

sentence_vectors = get_sentence_vectors(nlp, final_texts)



'''
# Using a MLP 

x_train, x_test, y_train, y_test = train_test_split(sentence_vectors, final_labels, shuffle=True, test_size=0.2)

# Converting to NumPy arrays
x_train = np.array(x_train).astype(np.float32)
x_test = np.array(x_test).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
y_test = np.array(y_test).astype(np.int32)


i = Input(shape=(x_train[0].shape))
x = Dense(units=512, activation='relu')(i)
x = Dropout(0.2)(x)
x = Dense(units=256, activation='relu')(x)   
x = Dropout(0.2)(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(0.2)(x)
o = Dense(units=1, activation='sigmoid')(x)
model = Model(i, o)


optimizer = SGD(learning_rate=0.01, momentum=0.4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])   
model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
'''


'''
# Using LSTM 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Model


y = np.array(labels).astype(np.int32)
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=50)

# Converting text into a sequence of integers based on the internal vocabulary  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


print('There are {} unique tokens'.format(len(tokenizer.word_index)))

embedding_size = 100
MAX_LEN = 50
VOCAB_SIZE = len(tokenizer.word_index)

x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post')
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post')

i = Input(shape=x_train[0].shape)

# Takes in sequences of integers and returns word vectors
x = Embedding(VOCAB_SIZE + 1, embedding_size)(i)
x = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
x = GlobalMaxPooling1D()(x)
o = Dense(1, activation='sigmoid')(x)

model = Model(i, o)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

'''

'''

# Using a CNN
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax


x_train, x_test, y_train, y_test = train_test_split(sentence_vectors, final_labels, shuffle=True, test_size=0.2)


x_train = np.array(x_train).astype(np.float32)
x_test = np.array(x_test).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
y_test = np.array(y_test).astype(np.int32)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)



i = Input(shape=(300,1))
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', strides=2)(i)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x)

x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', strides=2)(i)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x)

x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', strides=2)(i)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(0.2)(x)

o = Dense(units=1, activation='sigmoid')(x)

model = Model(i, o)


optimizer = Adamax(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])   
model.summary()

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

'''

# Predicting on the test set 
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import classification_report, confusion_matrix
cm = classification_report(y_test, y_pred)
print(cm)
conf = confusion_matrix(y_test, y_pred)
print(conf)


# Plotting the losses and accuracies
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()


# Predicting for a single tweet 
tweet = 'i am so depressed'
tweet_doc = nlp(tweet) 
tweet_vector = tweet_doc.vector

tweet_vector = np.reshape(tweet_vector, (1, 300))

y_pred = model.predict(tweet_vector)
print('The tweet was classified as {0:.2f}% as relating to depression'.format(y_pred[0][0]*100))