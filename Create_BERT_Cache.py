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
# Copyright 2019 José Moreno - Institut de Recherche en Informatique de Toulouse
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

from neuralnets.BERTWordEmbeddings import BERTWordEmbeddings
from util.CoNLL import readCoNLL
import os
import sys
import logging
import time

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


