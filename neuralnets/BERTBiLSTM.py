# Copyright 2018 Nils Reimers - Technical University of Darmstadt UKP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified version
# Copyright 2019-2020 José Moreno - Institut de Recherche en Informatique de Toulouse
#                     Luis Adrián Cabrera-Diego - La Rochelle Université
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging used for multi-task learning.

License: Apache-2.0
"""

from __future__ import print_function
from util import BIOF1Validation

import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import math
import numpy as np
import sys
import time
import os
import random
import logging
from tqdm import tqdm

from .keraslayers.ChainCRF import ChainCRF
from .keraslayers.WeightedAverage import WeightedAverage
from .BERTWordEmbeddings import BERTWordEmbeddings
from util import conllEVAL



class BERTBiLSTM:
    def __init__(self, embeddingsLookup, params=None):
        self.trainRangeLength = None
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.models = None
        self.modelSavePath = None
        self.resultsSavePath = None


        # Hyperparameters for the network
        defaultParams = {'dropout': (0.5,0.5), 'classifier': ['Softmax'], 'LSTM-Size': (100,), 'customClassifier': {},
                         'optimizer': 'adam',
                         'charEmbeddings': 'CNN', 'charEmbeddingsSize': 30, 'charFilterSize': 30, 'charFilterLength': 3, 'charLSTMSize': 25, 'maxCharLength': 25,
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 5, 'miniBatchSize': 32,
                         'featureNames': ['casing'], 'addFeatureDimensions': 10}
        if params != None:
            defaultParams.update(params)

        self.params = defaultParams

        self.embeddingsLookup = embeddingsLookup
        embConfig = self.embeddingsLookup.getConfig()
        self.params['embeddingsConfig'] = embConfig

        if embConfig['embeddings_path'] is not None and 'tokens_embeddings' not in self.params['featureNames']:
            self.params['featureNames'].append('tokens_embeddings')

        if embConfig['bert_mode'] is not None and 'bert_embeddings' not in self.params['featureNames']:
             self.params['featureNames'].append('bert_embeddings')
                   
        print(self.params)

    def setMappings(self, mappings):
        self.mappings = mappings

    def setDataset(self, datasets, data):
        self.datasets = datasets
        self.data = data

        # Create some helping variables
        self.mainModelName = None
        self.epoch = 0
        self.learning_rate_updates = {'sgd': {1: 0.1, 3: 0.05, 5: 0.01}}
        self.modelNames = list(self.datasets.keys())
        self.evaluateModelNames = []
        self.labelKeys = {}
        self.idx2Labels = {}
        self.trainMiniBatchRanges = None
        self.trainSentenceLengthRanges = None


        for modelName in self.modelNames:
            labelKey = self.datasets[modelName]['label']
            self.labelKeys[modelName] = labelKey
            self.idx2Labels[modelName] = {v: k for k, v in self.mappings[labelKey].items()}
            
            if self.datasets[modelName]['evaluate']:
                self.evaluateModelNames.append(modelName)
            
            logging.info("--- %s ---" % modelName)
            logging.info("%d train sentences" % len(self.data[modelName]['trainMatrix']))
            logging.info("%d dev sentences" % len(self.data[modelName]['devMatrix']))
            logging.info("%d test sentences" % len(self.data[modelName]['testMatrix']))



        
        if len(self.evaluateModelNames) == 1:
            self.mainModelName = self.evaluateModelNames[0]
             
        self.casing2Idx = self.mappings['casing']

        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            logging.info("Pad words to uniform length for characters embeddings")
            all_sentences = []
            for dataset in self.data.values():
                for data in [dataset['trainMatrix'], dataset['devMatrix'], dataset['testMatrix']]:
                    for sentence in data:
                        all_sentences.append(sentence)

            self.padCharacters(all_sentences)
            logging.info("Words padded to %d characters" % (self.maxCharLen))

        
    def buildModel(self):
        self.models = {}
        inputNodes = []
        mergeInputLayers = []
        outputEmbeddingFct = None
        #Embedding features
        for featureName in self.params['featureNames']:
            if featureName.endswith('_embeddings'):
                #sampleEmbedding = np.array(self.data[self.modelNames[0]]['trainMatrix'][0][featureName][0])
                if outputEmbeddingFct is None: #Compute a sample embedding to get the dimension
                    outputEmbeddingFct = self.embeddingsLookup.sentenceLookup([{'tokens': ['test']}])

                sampleEmbedding = np.asarray(outputEmbeddingFct[featureName.rsplit('_', 1)[0]][0][0])


                if len(sampleEmbedding.shape)==1:
                    tokens_input = Input(shape=(None, sampleEmbedding.shape[0]), dtype='float32',
                                         name=featureName + '_input')
                    inputNodes.append(tokens_input)
                    mergeInputLayers.append(tokens_input)
                elif len(sampleEmbedding.shape)==2:
                    bert_input = Input(shape=(None, sampleEmbedding.shape[0], sampleEmbedding.shape[1]),
                                       dtype='float32',
                                       name=featureName + '_input')
                    bert_output = TimeDistributed(WeightedAverage(), name=featureName + '_weighted_average')(
                        bert_input)

                    inputNodes.append(bert_input)
                    mergeInputLayers.append(bert_output)
                else:
                    print("Unknown embedding dimensions" + str(sampleEmbedding.shape))
                    assert (False)
            else:
                feature_input = Input(shape=(None,), dtype='int32', name=featureName+'_input')
                feature_embedding = Embedding(input_dim=len(self.mappings[featureName]), output_dim=self.params['addFeatureDimensions'], name=featureName+'_embeddings')(feature_input)

                inputNodes.append(feature_input)
                mergeInputLayers.append(feature_embedding)
        

        # :: Character Embeddings ::
        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            charset = self.mappings['characters']
            charEmbeddingsSize = self.params['charEmbeddingsSize']
            maxCharLen = self.maxCharLen
            charEmbeddings = []
            for _ in charset:
                limit = math.sqrt(3.0 / charEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, charEmbeddingsSize)
                charEmbeddings.append(vector)

            charEmbeddings[0] = np.zeros(charEmbeddingsSize)  # Zero padding
            charEmbeddings = np.asarray(charEmbeddings)

            chars_input = Input(shape=(None, maxCharLen), dtype='int32', name='char_input')
            mask_zero = (self.params['charEmbeddings'].lower()=='lstm') #Zero mask only works with LSTM
            chars = TimeDistributed(
                Embedding(input_dim=charEmbeddings.shape[0], output_dim=charEmbeddings.shape[1],
                          weights=[charEmbeddings],
                          trainable=True, mask_zero=mask_zero), name='char_emd')(chars_input)

            if self.params['charEmbeddings'].lower()=='lstm':  # Use LSTM for char embeddings from Lample et al., 2016
                charLSTMSize = self.params['charLSTMSize']
                chars = TimeDistributed(Bidirectional(LSTM(charLSTMSize, return_sequences=False)), name="char_lstm")(
                    chars)
            else:  # Use CNNs for character embeddings from Ma and Hovy, 2016
                charFilterSize = self.params['charFilterSize']
                charFilterLength = self.params['charFilterLength']
                chars = TimeDistributed(Conv1D(charFilterSize, charFilterLength, padding='same'), name="char_cnn")(
                    chars)
                chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)

            self.params['featureNames'].append('characters')
            mergeInputLayers.append(chars)
            inputNodes.append(chars_input)
            
        # :: Task Identifier :: 
        if self.params['useTaskIdentifier']:
            self.addTaskIdentifier()
            
            taskID_input = Input(shape=(None,), dtype='int32', name='task_id_input')
            taskIDMatrix = np.identity(len(self.modelNames), dtype='float32')
            taskID_outputlayer = Embedding(input_dim=taskIDMatrix.shape[0], output_dim=taskIDMatrix.shape[1], weights=[taskIDMatrix], trainable=False, name='task_id_embedding')(taskID_input)
        
            mergeInputLayers.append(taskID_outputlayer)
            inputNodes.append(taskID_input)
            self.params['featureNames'].append('taskID')

        if len(mergeInputLayers) >= 2:
            merged_input = concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]
        
        
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:      
            if isinstance(self.params['dropout'], (list, tuple)):  
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name='shared_varLSTM_'+str(cnt))(shared_layer)
            else:
                """ Naive dropout """
                shared_layer = Bidirectional(LSTM(size, return_sequences=True), name='shared_LSTM_'+str(cnt))(shared_layer) 
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)
            
            cnt += 1
            
            
        for modelName in self.modelNames:
            output = shared_layer
            
            modelClassifier = self.params['customClassifier'][modelName] if modelName in self.params['customClassifier'] else self.params['classifier']

            if not isinstance(modelClassifier, (tuple, list)):
                modelClassifier = [modelClassifier]
            
            cnt = 1
            for classifier in modelClassifier:
                n_class_labels = len(self.mappings[self.labelKeys[modelName]])

                if classifier == 'Softmax':
                    output = TimeDistributed(Dense(n_class_labels, activation='softmax'), name=modelName+'_softmax')(output)
                    lossFct = 'sparse_categorical_crossentropy'
                elif classifier == 'CRF':
                    output = TimeDistributed(Dense(n_class_labels, activation=None),
                                             name=modelName + '_hidden_lin_layer')(output)
                    crf = ChainCRF(name=modelName+'_crf')
                    output = crf(output)
                    lossFct = crf.sparse_loss
                elif isinstance(classifier, (list, tuple)) and classifier[0] == 'LSTM':
                            
                    size = classifier[1]
                    if isinstance(self.params['dropout'], (list, tuple)): 
                        output = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name=modelName+'_varLSTM_'+str(cnt))(output)
                    else:
                        """ Naive dropout """ 
                        output = Bidirectional(LSTM(size, return_sequences=True), name=modelName+'_LSTM_'+str(cnt))(output) 
                        if self.params['dropout'] > 0.0:
                            output = TimeDistributed(Dropout(self.params['dropout']), name=modelName+'_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(output)                    
                else:
                    assert(False) #Wrong classifier
                    
                cnt += 1
                
            # :: Parameters for the optimizer ::
            optimizerParams = {}
            if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
                optimizerParams['clipnorm'] = self.params['clipnorm']
            
            if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
                optimizerParams['clipvalue'] = self.params['clipvalue']
            
            if self.params['optimizer'].lower() == 'adam':
                opt = Adam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'nadam':
                opt = Nadam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'rmsprop': 
                opt = RMSprop(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adadelta':
                opt = Adadelta(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adagrad':
                opt = Adagrad(**optimizerParams)
            elif self.params['optimizer'].lower() == 'sgd':
                opt = SGD(lr=0.1, **optimizerParams)
            


            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=lossFct, optimizer=opt)
            
            model.summary(line_length=125)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))
            
            self.models[modelName] = model
        


    def trainModel(self):
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:       
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for modelName in self.modelNames:            
                K.set_value(self.models[modelName].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])

        for batch in tqdm(self.minibatch_iterate_dataset(), total=self.trainRangeLength, desc="Training", unit=' Batch'):
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                #print(nnInput[0].shape,nnInput[1].shape,nnInput[2].shape)
                self.models[modelName].train_on_batch(nnInput, nnLabels)


    def minibatch_iterate_dataset(self, modelNames = None):
        """ Create batches based on sentence length with the same size. Sentences and
        mini-batch chunks are shuffled and used to the train the model """
        
        if self.trainSentenceLengthRanges == None:
            """ Create mini batch ranges """
            self.trainSentenceLengthRanges = {}
            self.trainMiniBatchRanges = {}            
            for modelName in self.modelNames:
                trainData = self.data[modelName]['trainMatrix']
                trainData.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by sentence length
                trainRanges = []
                oldSentLength = len(trainData[0]['tokens'])            
                idxStart = 0
                
                #Find start and end of ranges with sentences with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])
                    
                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx))
                        idxStart = idx
                    
                    oldSentLength = sentLength
                
                #Add last sentence
                trainRanges.append((idxStart, len(trainData)))
                
                
                #Break up ranges into smaller mini batch sizes
                miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]

                    bins = int(math.ceil(rangeLen/float(self.params['miniBatchSize'])))
                    binSize = int(math.ceil(rangeLen / float(bins)))
                    
                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))
                      
                self.trainSentenceLengthRanges[modelName] = trainRanges
                self.trainMiniBatchRanges[modelName] = miniBatchRanges
                
        if modelNames == None:
            modelNames = self.modelNames
            
        #Shuffle training data
        for modelName in modelNames:      
            #1. Shuffle sentences that have the same length
            x = self.data[modelName]['trainMatrix']
            for dataRange in self.trainSentenceLengthRanges[modelName]:
                for i in reversed(range(dataRange[0]+1, dataRange[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(dataRange[0], i)
                    x[i], x[j] = x[j], x[i]
               
            #2. Shuffle the order of the mini batch ranges       
            random.shuffle(self.trainMiniBatchRanges[modelName])
     
        
        #Iterate over the mini batch ranges
        if self.mainModelName != None:
            self.trainRangeLength = len(self.trainMiniBatchRanges[self.mainModelName])
        else:
            self.trainRangeLength = min([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])

        

        for idx in range(self.trainRangeLength):
            batches = {}
            
            for modelName in modelNames:   
                trainMatrix = self.data[modelName]['trainMatrix']
                dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])] 
                labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                labels = np.expand_dims(labels, -1)
                batches[modelName] = [labels]
                batches[modelName].extend(self.getInputData(trainMatrix, range(dataRange[0], dataRange[1])))
            
            yield batches   
            

        
    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsSavePath = open(resultsFilepath, 'w')
        else:
            self.resultsSavePath = None
        
    def fit(self, epochs):
        if self.models is None:
            self.buildModel()

        total_train_time = 0
        max_dev_score = {modelName:0 for modelName in self.models.keys()}
        max_test_score = {modelName:0 for modelName in self.models.keys()}
        no_improvement_since = 0
        print(self.idx2Labels) 
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time() 
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
                
            start_time = time.time() 
            for modelName in self.evaluateModelNames:
                logging.info("-- %s --" % (modelName))
                dev_score, test_score = self.computeScore(modelName, self.data[modelName]['devMatrix'], self.data[modelName]['testMatrix'])
         
                
                if dev_score > max_dev_score[modelName]:
                    max_dev_score[modelName] = dev_score
                    max_test_score[modelName] = test_score
                    no_improvement_since = 0
                    self.bestpredLabels = self.newpredLabels

                    #Save the model
                    if self.modelSavePath != None:
                        self.saveModel(modelName, epoch, dev_score, test_score)
                else:
                    no_improvement_since += 1
                    
                    
                if self.resultsSavePath != None:
                    self.resultsSavePath.write("\t".join(map(str, [epoch + 1, modelName, dev_score, test_score, max_dev_score[modelName], max_test_score[modelName]])))
                    self.resultsSavePath.write("\n")
                    self.resultsSavePath.flush()
                
                logging.info("\nScores from epoch with best dev-scores:\n  Dev-Score: %.4f\n  Test-Score: %.4f" % (max_dev_score[modelName], max_test_score[modelName]))
                logging.info("")
                
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
            if self.params['earlyStopping']  > 0 and no_improvement_since >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break

    def tagSentences(self, sentences):
        # Pad characters
        if 'characters' in self.params['featureNames']:
            self.padCharacters(sentences)

        labels = {}
        for modelName, model in self.models.items():
            paddedPredLabels = self.predictLabels(model, sentences)
            predLabels = []
            total_sentences = len(sentences)
            processed_sentences = 0

            for (idx, sentence) in enumerate(sentences):
                unpaddedPredLabels = []
                for (tokenIdx, token) in enumerate(sentence['tokens']):
                    if token != 0:  # Skip padding tokens
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])
                predLabels.append(unpaddedPredLabels)
                processed_sentences += 1
                print(f"Processed sentences: {processed_sentences}/{total_sentences}", end="\r")
            print("\n")
            idx2Label = self.idx2Labels[modelName]
            labels[modelName] = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]

        return labels
            
    
    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)
        
        return sentenceLengths

    def predictLabels(self, model, sentences):
        predLabels = [None]*len(sentences)
        sentenceLengths = self.getSentenceLengths(sentences)
        for indices in sentenceLengths.values():   
            nnInput = self.getInputData(sentences, indices)
            predictions = model.predict(nnInput, verbose=False)
            predictions = predictions.argmax(axis=-1) #Predict classes            

            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx]    
                predIdx += 1   

        return predLabels

    def getInputData(self, sentences, indices):
        nnInput = []
        for featureName in self.params['featureNames']:
            if featureName.endswith('_embeddings'):
                pureFeatureName = featureName.rsplit('_', 1)[0]
                batch_sentences = [sentences[idx] for idx in indices]
                #print("##############",pureFeatureName)
                #print(batch_sentences)
                inputData = np.asarray(self.embeddingsLookup.batchLookup(batch_sentences, pureFeatureName))
                #print(inputData.shape)
                if inputData.shape[1]==1:  # One token sentence need to be padded for CRF
                    zeros = np.zeros(inputData.shape)
                    inputData = np.append(inputData, zeros, axis=1)
            else:
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
            nnInput.append(inputData)
        return nnInput
    
   
    def computeScore(self, modelName, devMatrix, testMatrix):
        if self.labelKeys[modelName].endswith('_BIO') or self.labelKeys[modelName].endswith('_IOBES') or self.labelKeys[modelName].endswith('_IOB'):
            return self.computeF1Scores(modelName, devMatrix, testMatrix)
        else:
            return self.computeAccScores(modelName, devMatrix, testMatrix)   

    def computeF1Scores(self, modelName, devMatrix, testMatrix):
        dev_pre, dev_rec, dev_f1 = self.computeF1(modelName, devMatrix)
        logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (dev_pre, dev_rec, dev_f1))
        
        test_pre, test_rec, test_f1 = self.computeF1(modelName, testMatrix, tosave=True)
        logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))
        
        return dev_f1, test_f1
    
    def computeAccScores(self, modelName, devMatrix, testMatrix):
        dev_acc = self.computeAcc(modelName, devMatrix)
        test_acc = self.computeAcc(modelName, testMatrix)
        
        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
        
        return dev_acc, test_acc   
        
        
    def computeF1(self, modelName, sentences, tosave=False):
        labelKey = self.labelKeys[modelName]
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]
        
        correctLabels = [sentences[idx][labelKey] for idx in range(len(sentences))]
        predLabels = self.predictLabels(model, sentences)
        if tosave:
          self.correctLabels = correctLabels
          self.newpredLabels = predLabels
          print([(y,idx2Label[y]) for x in predLabels for y in x][:50])

        labelKey = self.labelKeys[modelName]
        encodingScheme = labelKey[labelKey.index('_')+1:]
        
        pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'O', encodingScheme)
        pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'B', encodingScheme)
        
        if f1_b > f1:
            logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
            pre, rec, f1 = pre_b, rec_b, f1_b

        ###Added
        print(correctLabels[:5],'\n',predLabels[:5])
        correctLabels = [idx2Label[x] for y in correctLabels for x in y]
        predLabels = [idx2Label[x] for y in predLabels for x in y]
        print(correctLabels[:50],'\n',predLabels[:50])
        conllEVAL.evaluate(correctLabels, predLabels)
        ###EndAdded        
        return pre, rec, f1
    
    def computeAcc(self, modelName, sentences):
        correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        predLabels = self.predictLabels(self.models[modelName], sentences) 
        
        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1

  
        return numCorrLabels/float(numLabels)
    
    def padCharacters(self, sentences):
        """ Pads the character representations of the words to the longest word in the dataset """
        #Find the longest word in the dataset
        maxCharLen = self.params['maxCharLength']
        if maxCharLen <= 0:
            for sentence in sentences:
                for token in sentence['characters']:
                    maxCharLen = max(maxCharLen, len(token))
          

        for sentenceIdx in range(len(sentences)):
            for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
                token = sentences[sentenceIdx]['characters'][tokenIdx]

                if len(token) < maxCharLen: #Token shorter than maxCharLen -> pad token
                    sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')
                else: #Token longer than maxCharLen -> truncate token
                    sentences[sentenceIdx]['characters'][tokenIdx] = token[0:maxCharLen]
    
        self.maxCharLen = maxCharLen
        
    def addTaskIdentifier(self):
        """ Adds an identifier to every token, which identifies the task the token stems from """
        taskID = 0
        for modelName in self.modelNames:
            dataset = self.data[modelName]
            for dataName in ['trainMatrix', 'devMatrix', 'testMatrix']:            
                for sentenceIdx in range(len(dataset[dataName])):
                    dataset[dataName][sentenceIdx]['taskID'] = [taskID] * len(dataset[dataName][sentenceIdx]['tokens'])
            
            taskID += 1


    def saveModel(self, modelName, epoch, dev_score, test_score):
        import json
        import h5py

        if self.modelSavePath == None:
            raise ValueError('modelSavePath not specified.')

        savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch+1)).replace("[ModelName]", modelName)

        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info("Model "+savePath+" already exists. Model will be overwritten")

        self.models[modelName].save(savePath, True)
        with open(savePath+'_labelsoutput.txt', 'w') as f:
           for _list in self.bestpredLabels:
                f.write(" ".join([str(x) for x in _list]) + '\n') 

        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['mappings'] = json.dumps(self.mappings)
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['modelName'] = modelName
            h5file.attrs['labelKey'] = self.datasets[modelName]['label']
            #h5file.attrs['bestDev'] = self.predLabels




    @staticmethod
    def loadModel(modelPath, bert_path_name, bert_cuda_device, embeddings_file, use_fastext = False):
        import h5py
        import json
        from .keraslayers.ChainCRF import create_custom_objects

        f = h5py.File(modelPath, 'r')
        if keras.__version__ != f.attrs.get('keras_version').decode('utf-8'):
            print("The model was trained with a different version of Keras! " + f.attrs.get('keras_version').decode('utf-8') + " vs. " + keras.__version__)
            print("The model might break")

        custom_layers = create_custom_objects()
        custom_layers['WeightedAverage'] = WeightedAverage
        model = keras.models.load_model(modelPath, custom_objects=custom_layers)

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']

        # :: Read the config values for the embedding lookup function ::
        embeddings_path = params['embeddingsConfig']['embeddings_path']     
        bert_mode = params['embeddingsConfig']['bert_mode']
        #The location of the berth models could have changed and the configuration might be different
        if bert_path_name == "":
            bert_path_name = params['embeddingsConfig']['bert_path']
        bert_n_layers = params['embeddingsConfig']['bert_n_layers'] if 'bert_n_layers' in params['embeddingsConfig'] else None
        if bert_cuda_device == "":
            bert_cuda_device = -1  # Which GPU to use. -1 for CPU
    
        embLookup = BERTWordEmbeddings(embeddings_file, use_fastext, bert_path_name, bert_mode=bert_mode, bert_cuda_device=bert_cuda_device, bert_n_layers=bert_n_layers) if not bert_n_layers is None else BERTWordEmbeddings(embeddings_file, bert_path_name, bert_mode=bert_mode, bert_cuda_device=bert_cuda_device)

        # :: Create new model object ::
        bilstm = BERTBiLSTM(embLookup, params)
        bilstm.setMappings(mappings)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        bilstm.idx2Labels = {}
        bilstm.idx2Labels[modelName] = {v: k for k, v in bilstm.mappings[labelKey].items()}
        return bilstm
