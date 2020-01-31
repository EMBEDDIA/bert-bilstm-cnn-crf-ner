import urllib.request as urllib2
import urllib.parse as urlparse
from urllib.request import urlretrieve
import logging
import numpy as np
import pickle as pkl
import os
import gzip
import sys

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from fasttext import load_model

class MyBertModel(BertModel):
    def __init__(self, bert_path, bert_n_layers):
        BertModel.__init__(self, bert_path)
        self.bert_n_layers = bert_n_layers

    def checksentence(self,toksxtok,berttokens):
        l=[]
        for x in toksxtok:
          if x > 0:
            l.append(0)
            l.extend([1]*(x-1))
        for i in range(len(berttokens)):
            if l[i]>0 and (not berttokens[i].startswith("##")):
                 berttokens[i] = "##"+berttokens[i]            
        return berttokens

    def replaceUNKtokens(self,_sentence):
        ntokens_per_token = [len(self.berttok.tokenize(x)) for x in _sentence]
        if len([x for x in ntokens_per_token if x==0])==0:
           return _sentence
        randTok = None
        for i in range(-1,-len(ntokens_per_token),-1):
          if randTok == None and ntokens_per_token[i] > 0:
            randTok = _sentence[i]
        for i,x in enumerate(ntokens_per_token):
          if x == 0:
            _sentence[i] = randTok
        return _sentence

    def embed_sentences(self,sentences):
        emb = []
        max_size = 512
        used_layers = self.bert_n_layers
        for _sentence in sentences:
            _sentence = self.replaceUNKtokens(_sentence)
            ntokens_per_token = [len(self.berttok.tokenize(x)) for x in _sentence]
            sentence = "[CLS] "+" ".join(_sentence)+" [SEP]"
            tokenized_text = self.berttok.tokenize(sentence)#[:max_size]
            indexed_tokens = self.berttok.convert_tokens_to_ids(tokenized_text)
            tokenized_text = self.checksentence(ntokens_per_token,tokenized_text[1:-1])
            tokens_tensor = torch.tensor([indexed_tokens[:max_size]])
            res = self(tokens_tensor)
            resl = [res[0][ii][0] for ii in range(len(res[0]))]                
            for i in range(int(len(indexed_tokens)/max_size)):
                tokens_tensor = torch.tensor([indexed_tokens[max_size*(i+1):max_size*(i+2)]])
                res_temp = self(tokens_tensor)
                res_templ = [res_temp[0][ii][0] for ii in range(len(res_temp[0]))]
                resl = [torch.cat((x1,x2),0) for x1,x2 in zip(resl,res_templ)]
            idx = len(tokenized_text)-2
            embtokLayers = [[] for x in range(used_layers)]
            embtok = [[] for x in range(used_layers)]
            for tok in tokenized_text:
                for i in range(used_layers):
                    embtok[i].append(resl[i][idx].cpu().detach().numpy())
                idx -= 1
                if tok.startswith("##"):
                   continue
                for i in range(used_layers):
                    embtokLayers[i] = [np.average(embtok[i],axis=0).tolist()] + embtokLayers[i]#([res[0][x][0][idx] for x in range(12)])
                embtok = [[] for x in range(used_layers)]
            emb.append(embtokLayers)
        return emb


class BERTWordEmbeddings:
    def __init__(self, embeddings_path, use_fasttext, bert_path, bert_n_layers=12, bert_mode='average', bert_cuda_device=-1):
        self.embeddings_path = embeddings_path
        self.embedding_name = os.path.splitext(os.path.basename(embeddings_path))[0] if embeddings_path is not None else 'None'
        self.word2Idx = None
        self.embeddings = None
        self.ftmodel = {}

        self.bert_cuda_device=bert_cuda_device
        self.bert_path = bert_path
        self.bert_n_layers = bert_n_layers

        self.bert_mode = bert_mode
        self.bert = None

        self.cache_computed_bert_embeddings = False
        self.cache = {}
        self.lazyCacheFiles = []

        self.use_fastext = use_fasttext

        if self.embeddings_path is not None:
            self.__loadEmbeddings()

    def __loadEmbeddings(self):
        if self.use_fastext:
            self.ftmodel = load_model(self.embeddings_path)
        else:
            self.word2Idx, self.embeddings = self.readEmbeddings(self.embeddings_path)

    def getConfig(self):
        return {
            "embeddings_path": self.embeddings_path,
            "bert_mode": self.bert_mode,
            "bert_path": self.bert_path,
            "bert_n_layers": self.bert_n_layers,
            "bert_cuda_device": self.bert_cuda_device
        }

    def sentenceLookup(self, sentences):
        bert_vectors = None

        # :: BERT ::
        if self.bert_mode is not None:
            bert_vectors = self.getBertEmbedding(sentences)

        # :: Word Embedding ::
        tokens_vectors = None
        if self.embeddings_path is not None:
            tokens_vectors = []
            oov = []
            for sentence in sentences:
                per_token_embedding = []
                for token in sentence['tokens']:
                    if self.use_fastext:
                        vecVal = self.ftmodel.get_word_vector(token)
                    else:
                        vecId = self.word2Idx['UNKNOWN_TOKEN']
                        vecVal = self.embeddings[vecId]
                        if token in self.word2Idx:
                            vecId = self.word2Idx[token]
                            vecVal = self.embeddings[vecId]
                        elif token.lower() in self.word2Idx:
                            vecId = self.word2Idx[token.lower()]
                            vecVal = self.embeddings[vecId]
                        else:
                            oov.append(token)
                    per_token_embedding.append(vecVal if self.use_fastext else self.embeddings[vecId])
                per_token_embedding = np.asarray(per_token_embedding)
                tokens_vectors.append(per_token_embedding)
        out_vectors = {}
        if tokens_vectors is not None:
            out_vectors['tokens'] = tokens_vectors

        if bert_vectors is not None:
            out_vectors['bert'] = bert_vectors

        return out_vectors

    def batchLookup(self, sentences, feature_name):
        if feature_name == 'tokens':
            tokens_vectors = []
            oov = []
            for sentence in sentences:
                per_token_embedding = []
                for token in sentence['tokens']:
                    if self.use_fastext:
                        vecVal = self.ftmodel.get_word_vector(token)
                    else:
                        vecId = self.word2Idx['UNKNOWN_TOKEN']
                        vecVal = self.embeddings[vecId]
                        if token in self.word2Idx:
                            vecId = self.word2Idx[token]
                            vecVal = self.embeddings[vecId]
                        elif token.lower() in self.word2Idx:
                            vecId = self.word2Idx[token.lower()]
                            vecVal = self.embeddings[vecId]
                        else:
                            oov.append(token)
                    per_token_embedding.append(vecVal)
                per_token_embedding = np.asarray(per_token_embedding)
                tokens_vectors.append(per_token_embedding)
            return np.asarray(tokens_vectors)
        elif feature_name == 'bert':
            return np.asarray(self.getBertEmbedding(sentences))
        else:
            print("Unknown feature name was passed to singleSentenceLookup")
            assert(False)

    def applyBertMode(self, bert_vectors):
        if self.bert_mode == 'average':
            return np.average(bert_vectors, axis=0).astype(np.float32)
        elif self.bert_mode == 'weighted_average':
            return np.swapaxes(bert_vectors,0,1)
        elif self.bert_mode == 'last':
            return bert_vectors[-1, :, :]
        elif isinstance(self.bert_mode, int):
            return bert_vectors[int(self.bert_mode), :, :]
        else:
            print("Unknown BERT mode")
            assert (False)

    def getBertEmbedding(self, sentences):
        if len(self.lazyCacheFiles) > 0:
            self._loadLazyCache()

        bert_embeddings = []
        non_cached_sentences = []
        non_cached_sentences_indices = []

        # :: Lookup cached sentences ::
        for sentence in sentences:
            tokens = sentence['tokens']
            cache_key = tuple(tokens)
            if len(self.cache) > 0 and cache_key in self.cache:
                bert_embeddings.append(self.applyBertMode(self.cache[cache_key]))
            else:
                non_cached_sentences.append(tokens)
                non_cached_sentences_indices.append(len(bert_embeddings))
                bert_embeddings.append(None)

        # :: Compute BERT on the fly ::
        if len(non_cached_sentences) > 0:
            #print("On the fly ",len(non_cached_sentences))
            if self.bert is None:
                self.loadBERT()

            idx = 0
            for bert_vectors in self.bert.embed_sentences(non_cached_sentences):
                bert_embeddings[non_cached_sentences_indices[idx]] = self.applyBertMode(bert_vectors)
                if self.cache_computed_bert_embeddings:
                    tokens = non_cached_sentences[idx]
                    cache_key = tuple(tokens)
                    self.cache[cache_key] = bert_vectors

                idx += 1

        return bert_embeddings

    def getIdentifier(self):
        """Returns a unique identifier for this lookup function"""
        return "BERTWordEmbeddings_" + self.embedding_name + "_" + str(self.bert_mode)

    def loadBERT(self):
        self.bert = MyBertModel.from_pretrained(self.bert_path, self.bert_n_layers)
        self.bert.berttok = BertTokenizer.from_pretrained(self.bert_path)

    def loadCache(self, inputPath):
        self.lazyCacheFiles.append(inputPath)

    def storeCache(self, outputPath):
        f = open(outputPath, 'wb')
        pkl.dump(self.cache, f, -1)
        f.close()

    def addToCache(self, sentences):
        if self.bert is None:
            self.loadBERT()

        idx = 0
        for bertEmbedding in self.bert.embed_sentences(sentences):
            cache_key = tuple(sentences[idx])
            self.cache[cache_key] = bertEmbedding

            idx += 1

    def _loadLazyCache(self):
        while len(self.lazyCacheFiles) > 0:
            inputPath = self.lazyCacheFiles.pop()

            if not os.path.isfile(inputPath):
                print("BERT cache file not found:", inputPath)
                continue

            f = open(inputPath, 'rb')
            loaded_cache = pkl.load(f)
            f.close()

            if len(self.cache) == 0:
                self.cache = loaded_cache
            else:
                self.cache.update(loaded_cache)

    def readEmbeddings(self, embeddingsPath):
        filename = os.path.basename(embeddingsPath)
        if not os.path.isfile(embeddingsPath):
            if filename in ['komninos_english_embeddings.gz', 'levy_english_dependency_embeddings.gz',
                            'reimers_german_embeddings.gz']:
                self.getEmbeddings(filename, embeddingsPath)
            else:
                print("The embeddings file %s was not found" % embeddingsPath)
                exit()

        # :: Read in word embeddings ::
        logging.info("Read file: %s" % embeddingsPath)
        word2Idx = {}
        embeddings = []
        embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
        embeddingsDimension = None

        for line in embeddingsIn:
            split = line.rstrip().split(" ")
            word = split[0]

            if embeddingsDimension == None:
                embeddingsDimension = len(split) - 1
                if embeddingsDimension == 1:
                    embeddingsDimension = None
                    continue


            if (len(split) - 1)!=embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
                print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                continue


            if len(word2Idx)==0:  # Add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(embeddingsDimension)
                embeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                rndState = np.random.RandomState(
                    seed=12345)  # Fixed rnd seed for unknown token, so that it is always the same
                vector = rndState.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)

                embeddings.append(vector)

            if len(split)<=2 or word in word2Idx:
                continue

            if embeddingsDimension==None:
                embeddingsDimension = len(split) - 1

            if (len(
                    split) - 1)!=embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
                print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                continue


            vector = np.array([float(num) for num in split[1:]])

            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)
        logging.info("Vocab size: %s" % str(len(word2Idx)))

        return word2Idx, embeddings

    def getEmbeddings(self, filename, savePath):
        if not os.path.isfile(savePath):
            self.download("https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/" + filename, savePath)

    def download(self, url, savePath, silent=False):
        filename = os.path.basename(urlparse.urlparse(url).path) or 'downloaded.file'

        def get_size():
            meta = urllib2.urlopen(url).info()
            meta_func = meta.getheaders if hasattr(
                meta, 'getheaders') else meta.get_all
            meta_length = meta_func('Content-Length')
            try:
                return int(meta_length[0])
            except:
                return 0

        def kb_to_mb(kb):
            return kb / 1024.0 / 1024.0

        def callback(blocks, block_size, total_size):
            current = blocks * block_size
            percent = 100.0 * current / total_size
            line = '[{0}{1}]'.format(
                '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
            status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
            sys.stdout.write(
                status.format(
                    percent, line, kb_to_mb(current), kb_to_mb(total_size)))

        logging.info(
            'Downloading: {0} ({1:3.1f} MB)'.format(url, kb_to_mb(get_size())))
        try:
            (savePath, headers) = urlretrieve(url, savePath, None if silent else callback)
        except:
            os.remove(savePath)
            raise Exception("Can't download {0}".format(savePath))
        else:
            print()
            logging.info('Downloaded to: {0}'.format(savePath))

        return savePath

