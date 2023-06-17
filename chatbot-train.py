import random
import json
import pickle

# Machine Learning libraries
import numpy as np
import tensorflow as tf
import nltk

# Lemmatization = Going back to its root word
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Open the intents.json file to read
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']  # Ignores unnecessary characters

# Intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# TRAINING
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_X = training[:, :len(words)]
train_Y = training[:, len(words):]

# Creating the LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(words), 128, input_length=len(train_X[0])))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_Y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Final part of training neural network
hist = model.fit(np.array(train_X), np.array(train_Y), epochs=250, batch_size=5, verbose=1)
model.save('axel_brainmodel.h5', hist)  # Saves the training model as a file
print('Sensei: Training is done!')
