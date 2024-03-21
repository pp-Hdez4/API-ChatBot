from django.shortcuts import render
from django.http import JsonResponse
import os
import random
import os
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Create your views here
def home(request):
    return render(request, 'core/chat.html')

# Obtén la ruta del directorio actual del script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Actualiza la carga del archivo intents.json
intents_path = os.path.join(current_dir, 'intents.json')
with open(intents_path, encoding='utf-8') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
words = pickle.load(open(os.path.join(current_dir, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(current_dir, 'classes.pkl'), 'rb'))
model = load_model(os.path.join(current_dir, 'chatbot_model.h5'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def get_response(user_message, intents_json):
    responses = []
    for intent in intents_json['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in user_message.lower():
                responses.extend(intent['responses'])
                break
        if responses:
            return random.choice(responses)
    return "Lo siento, no tengo información sobre este tema :("

def getResponse(request):
    if request.method == 'GET':
        user_message = request.GET.get('userMessage')
        print("Solicitud recibida. Mensaje del usuario:", user_message)
        # Obtener la respuesta basada en el mensaje del usuario
        response = get_response(user_message, intents)
        print("Respuesta generada:", response)
        return JsonResponse({'botResponse': response})
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)
