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

from __future__ import print_function
import os
import argparse
import logging
import sys
import torch
from neuralnets.BERTBiLSTM import BERTBiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from neuralnets.BERTWordEmbeddings import BERTWordEmbeddings
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
sess = tf.Session(config=config)
K.set_session(sess)


##################################################

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



######################################################
#
# Global Parameters
#
######################################################
parser = argparse.ArgumentParser()
parser.add_argument("--bert_n_layers", type=int, default=2)
#parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--n_epochs", type=int, default=25)
#parser.add_argument("--finetuning", dest="finetuning", action="store_true")
#parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
parser.add_argument("--jobid", type=str, default="282NER_ru")
parser.add_argument("--dataset_name", type=str, default="282NER_ru")
parser.add_argument("--tagging_format", type=str, default="NER_IOB")
parser.add_argument("--embeddings_file", type=str, default="/home/adrian/Programs/bert-bilstm-cnn-crf-ner/embeddings/fastText157/cc.ru.300.vec.gz.top1.bin")
parser.add_argument("--bert_path", type=str, default="bert-base-multilingual-cased")
#parser.add_argument("--bert_path", type=str, default="/home/adrian/Programs/bert-bilstm-cnn-crf-ner/bert/shebert-pytorch/")
hp = parser.parse_args()
print("hyperparameters ",hp)

dataset_name = hp.dataset_name
embeddings_file = hp.embeddings_file#'embeddings/komninos_english_embeddings.gz'#'embeddings/fastText157/cc.bg.300.vec.gz.top1.gz'#'embeddings/fastTextMulti/wiki.multi.en.vec.gz'#'embeddings/fastText157/cc.pl.300.vec.gz'#'embeddings/komninos_english_embeddings.gz'
bert_path = hp.bert_path
jobid = hp.jobid
nepochs = hp.n_epochs
bert_n_layers = hp.bert_n_layers
tagging_format = hp.tagging_format

######################################################
#
# Data preprocessing
#
######################################################

datasets = {
    dataset_name :                                   #Name of the dataset
        {'columns': {1:'tokens', 3:'NER_IOB'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': tagging_format,                              #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}


# :: Transform datasets to a pickle file ::
pickleFile = perpareDataset(datasets)

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
bert_mode = 'weighted_average'

#Which GPU to use for . -1 for CPU
if torch.cuda.is_available():
    print("Using CUDA")
    bert_cuda_device = 0
else:
    print("Using CPU")
    bert_cuda_device = -1

embLookup = BERTWordEmbeddings(embeddings_file, True, bert_path, bert_n_layers=bert_n_layers, bert_cuda_device=bert_cuda_device)

# You can use a cache that stores the computed BERT embeddings.
# This increases the training speed, as BERT embeddings need to computed only once.
# However, it leads to a significant memory overhead of multiple GB (requires about 24KB per token).
embLookup.cache_computed_bert_embeddings = True

# We can add a pre-computed BERT cache to the class. See Create_BERT_Cache.py how to pre-compute such a cache.
bertfn = bert_path.split('/')
bertfn = bertfn[-1] if len(bertfn[-1]) > 1 else bertfn[-2]
embLookup.loadCache('embeddings/bert_'+bertfn+'_cache_'+dataset_name+'.pkl')





######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.5, 0.5)}

model = BERTBiLSTM(embLookup, params)
model.setMappings(mappings)
model.setDataset(datasets, data)
model.modelSavePath = "models/"+jobid+"/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=nepochs)



