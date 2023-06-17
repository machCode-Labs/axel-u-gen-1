# Birthdate = JUNE 8, 2023
# For the AI, find more corpus text in Gutenberg website or more to train the model to speak.

import random
import json
import pickle
import time

# Machine Learning libraries
import numpy as np
import nltk

# Speech Recognition
import speech_recognition as sr
recognizer = sr.Recognizer()

# TTS
import pyttsx3
engine = pyttsx3.init()

engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)

# Lemmatization = Going back to its root word
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from keras.models import load_model # For loading the model
lemmatizer = WordNetLemmatizer()

# # GPT-2 to make text more human. (This works by the way... it just doesn't stop if sentence is near finished...)
# from transformers import pipeline
# gpt2_generator = pipeline('text-generation', model='gpt2')

intents = json.loads(open(r'C:/Users/Mark James/Desktop/AXEL-U/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('axel_brainmodel.h5') # The AI model created in the new.py

# All new responses and inputs will be used for training the model in new.py
queryFile = 'C:/Users/Mark James/Desktop/AXEL-U/data/queries.txt' # Text file for storing user inputs

# The Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])

            text = result
            engine.say(text)
            engine.runAndWait()

            # sentence = gpt2_generator(result, do_sample=True, top_k=50, early_stopping=True, temperature=1, max_length=100, no_repeat_ngram_size=2, num_beams=5)
            print("Predicted: " + i['tag'])
    
    return result

# Print to signal if bot is online
print("AXEL-U is now running. Conversation is active.")

# Create animation similar to chatGPT
def print_text_animated(text):
    words = text.split()
    for word in words:
        print(word, end = " ", flush = True)
        time.sleep(0.1)
        # print("\n======\n")

# Responsible for speech-to-text.
def speech_to_text(messagetext):
    # Create a recognizer object
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")

        recognizer.adjust_for_ambient_noise(source)

        # Capture the audio input from the microphone
        audio = recognizer.listen(source)

        try:
            # Converts speech to text.
            messagetext = recognizer.recognize_google(audio)
            print("You said: ", messagetext)

        # Fallbacks
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Error: {0}".format(e))

    return messagetext

while True:
    # message = input("") Use this if you want to use text.
    message = speech_to_text(input) # Uses microphone
    ints = predict_class(message)

    # User inputs (aka Patterns or Query) will be used for pretraining the model.
    with open (queryFile, 'a') as queryLine:
        queryLine.write("\"" + message + '\"\n')
    
    res = get_response(ints, intents)
    # print(res)
    # print_text_animated(res) If GPT-2 is enabled
    print_text_animated((res + "\n") + "======" * 3 +  ("\n"))