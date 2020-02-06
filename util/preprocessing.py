from __future__ import (division, absolute_import, print_function, unicode_literals)
import os
import numpy as np
import os.path
import logging
from .CoNLL import readCoNLL

import sys
import pickle as pkl

from unidecode import unidecode

def perpareDataset(datasets, padOneTokenSentence=True):
    pklName = "_".join(sorted(datasets.keys()))
    outputPath = 'pkl/' + pklName + '.pkl'

    if os.path.isfile(outputPath):
        logging.info("Using existent pickle file: %s" % outputPath)
        return outputPath

    casing2Idx = getCasingVocab()

    mappings = {'tokens': {}, 'casing': casing2Idx}
    pklObjects = {'mappings': mappings, 'datasets': datasets, 'data': {}}

    for datasetName, dataset in datasets.items():
        datasetColumns = dataset['columns']
        commentSymbol = dataset['commentSymbol']

        trainData = '/home/adrian/DataSets/NER/%s/train.txt' % datasetName
        devData = '/home/adrian/DataSets/NER/%s/dev.txt' % datasetName
        testData = '/home/adrian/DataSets/NER/%s/test.txt' % datasetName
        paths = [trainData, devData, testData]

        logging.info("\n:: Transform "+datasetName+" dataset ::")
        pklObjects['data'][datasetName] = createPklFiles(paths, mappings, datasetColumns, commentSymbol, padOneTokenSentence)


    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("\n\nDONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath


def loadDatasetPickle(embeddingsPickle):
    """ Loads the cPickle file, that contains the word embeddings and the datasets """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()

    return pklObjects['mappings'], pklObjects['data']


def addCharAndCasingInformation(sentences):
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)
            sentences[sentenceIdx]['casing'].append(getCasing(token))


def addCharInformation(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)


def addCasingInformation(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))
       
       
def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def getCasingVocab():
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}


def createMatrices(sentences, mappings, padOneTokenSentence):
    #symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")
    #tr = {ord(a):ord(b) for a, b in zip(*symbols)}
    data = []
    total_sentences = len(sentences)
    processed_sentences = 0
    for sentence in sentences:
        row = {name: [] for name in list(mappings.keys())}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
                    
            for entry in sentence[mapping]:                
                if mapping.lower() == 'tokens':
                    idx = entry
                elif mapping.lower() == 'characters':
                    entry = [x for c in entry for x in unidecode(c)]#entry.translate(tr)  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])                           
                                      
                else:
                    idx = str2Idx[entry]
                                    
                row[mapping].append(idx)



        if len(row['tokens'])==1 and padOneTokenSentence:
            for mapping, str2Idx in mappings.items():
                if mapping.lower()=='tokens':
                    pass
                elif mapping.lower()=='characters':
                    row['characters'].append([0])
                else:
                    row[mapping].append(0)

        data.append(row)
        processed_sentences += 1
        print(f"Sentences appended to the matrix: {processed_sentences}/{total_sentences}", end="\r")
    print("\n")
    return data
    
  
  
def createPklFiles(datasetFiles, mappings, cols, commentSymbol, padOneTokenSentence):
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol)
   
    extendMappings(mappings, trainSentences+devSentences+testSentences)


    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset
    
    addCharInformation(trainSentences)
    addCasingInformation(trainSentences)
    
    addCharInformation(devSentences)
    addCasingInformation(devSentences)
    
    addCharInformation(testSentences)   
    addCasingInformation(testSentences)

    logging.info(":: Create Train Matrix ::")
    trainMatrix = createMatrices(trainSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Dev Matrix ::")
    devMatrix = createMatrices(devSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Test Matrix ::")
    testMatrix = createMatrices(testSentences, mappings, padOneTokenSentence)

    
    data = {
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def extendMappings(mappings, sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens') #No need to map tokens

    for sentence in sentences:
        for name in sentenceKeys:
            if name not in mappings:
                mappings[name] = {'O':0} #'O' is also used for padding

            for item in sentence[name]:              
                if item not in mappings[name]:
                    mappings[name][item] = len(mappings[name])
