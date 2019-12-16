from neuralnets.BERTWordEmbeddings import BERTWordEmbeddings
from util.CoNLL import readCoNLL
import os
import sys
import logging
import time
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

if len(sys.argv) < 3:
    print("Usage: python Create_BERT_Cache.py datasetName tokenColumnId [cuda_device]")
    exit()

datasetName = sys.argv[1]
tokenColId = int(sys.argv[2])
cudaDevice = int(sys.argv[3]) if len(sys.argv) >= 4 else -1


# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


commentSymbol = None
columns = {tokenColId: 'tokens'}



picklePath = "embeddings/bert_cache_" + datasetName + ".pkl"

bert_path = '/data6T/Datasets/BERT/cased_L-12_H-768_A-12/'
embLookup = BERTWordEmbeddings(None, bert_path)

print("Bert Cache Generation")
print("Output file:", picklePath)
print("CUDA Device:", cudaDevice)

splitFiles = ['train.txt', 'dev.txt', 'test.txt']

for splitFile in splitFiles:
    inputPath = os.path.join('data', datasetName, splitFile)

    print("Adding file to cache: "+inputPath)
    sentences = readCoNLL(inputPath, columns, commentSymbol)
    tokens = [sentence['tokens'] for sentence in sentences]

    start_time = time.time()
    embLookup.addToCache(tokens)
    end_time = time.time()
    print("%s processed in %.1f seconds" % (splitFile, end_time - start_time))
    print("\n---\n")

print("Store file at:", picklePath)
embLookup.storeCache(picklePath)


