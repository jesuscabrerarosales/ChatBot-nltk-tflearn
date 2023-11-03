import nltk as n
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow as tf

import json
import random
import pickle


#DESCOMENTAR POR PRIMERA VEZ AL EJECUTAR
#n.download('punkt')

with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

#MOSTRAR DATOS ALMACENADOS EN EL ARCHIVO JSON
#print(datos)

#CREAMOS UN ARCHIVO PARA TENER ALMACENADO EL CONOCIMIENTO

try:
    with open("variables.pickle","rb") as archivoPickle:
        palabras,tags,entrenamiento,salida = pickle.load(archivoPickle)

except:
        
    palabras=[]
    tags=[]
    auxX=[]
    auxY=[]
    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = n.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

        #MOSTRAR PALABRAS SEPARADAS PARA TOKENIZAR LAS PALABRAS Y OBTENER LAS PALABRAS CLAVE
    #print(palabras)
    #print(auxX)
    #print(auxY)
    #print(tags)



    #EN LA VARIABLE "w" se puede agregar la funcion lower() para hacer que las oraciones
        #partidas en palabras se puedan contar como igual en el segundo for anidado

    palabras = [stemmer.stem(w) for w in palabras if w != "?"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)
    entrenamiento = []
    salida = []
    salidaVacia=[0 for _ in range(len(tags))]


    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra=[stemmer.stem(w) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida=salidaVacia[:]
        filaSalida[tags.index(auxY[x])]=1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

        #PROBANDO LISTAS ALMACENADAS PARA EL ENTRENAMIENTO

        #print(entrenamiento)
        #print(salida)
    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)

    with open("variables.pickle","wb") as archivoPickle:
        pickle.dump((palabras,tags,entrenamiento,salida),archivoPickle)

tf.compat.v1.reset_default_graph()

#CREACION DE NEURONAS
red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,len(salida[0]), activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

try:
    modelo.load("modelo.tflearn")
except:   
    modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=10,show_metric=True)
    modelo.save("modelo.tflearn")




def mainChatBot():
    while True:
        #print("BIENVENIDO AL TEST")
        entrada =input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = n.word_tokenize(entrada)
        
        
        #####ERROR#####
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        #####ERROR#####


        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra==palabraIndividual:
                    cubeta[i]= 1


        #PORCENTAJE DE PROXIMIDAD DE TAG
        resultado = modelo.predict([numpy.array(cubeta)])
        #print(resultado)
        resultadoIndice = numpy.argmax(resultado)
        tag = tags[resultadoIndice]
        for tagAux in datos["contenido"]:
            if tagAux["tag"]==tag:
                respuesta = tagAux["respuestas"]
        print("CHATBOT: ",random.choice(respuesta))

mainChatBot()