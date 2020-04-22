# Embeddia Project - Named Entity Recognition
# Copyright © 2020 Luis Adrián Cabrera Diego - La Rochelle Université
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import glob
import os
import subprocess
import sys

import torch

import xml.sax

from neuralnets.BERTBiLSTM import BERTBiLSTM
from util.BIOF1Validation import computeMetrics
from util.CoNLL import readCoNLL
from util.XMLParser import parseXML
from util.preprocessing import createMatrices, addCharAndCasingInformation


def main_command():
    arguments_parser = argparse.ArgumentParser()
    required_arguments = arguments_parser.add_argument_group('required arguments')
    required_arguments.add_argument("-f", "--file", help="File to process", required=True)
    required_arguments.add_argument("-l", "--language", help="Language of the file to process (en|hr)", required=True)

    optional_arguments = arguments_parser.add_argument_group('optional arguments')
    optional_arguments.add_argument("-F", "--fileFormat", help="Format of input file [plain, tokens, IOB, XML_tokenized, XML_sentences, XML_Large_tokenized, XML_Large_sentences]. Default plain", default="plain")
    optional_arguments.add_argument("-o", "--outputFile", help="Output file", default="")
    optional_arguments.add_argument("-H", "--HTMLFormat", help="Output file as HTML", action="store_true")
    optional_arguments.add_argument("-e", "--evaluate", help="Evaluate (only valid with IOB files)", action="store_true")
    optional_arguments.add_argument("-s", "--lastSentence", help="ID of the last Sentence Processed. Valid only for large XML.", type=int, default=-1)

    arguments = arguments_parser.parse_args()

    if arguments.outputFile == "":
        if not os.path.exists("./results"):
            os.mkdir("./results")
        arguments.outputFile = f"./results/{arguments.language}.html"

    print(f"File to process: {arguments.file}")
    print(f"Language to process: {arguments.language}")
    print(f"Output redirected to:", {arguments.outputFile})

    if arguments.lastSentence > -1:
        output_file = open(arguments.outputFile, "a")
        log_file = open(arguments.outputFile + ".log", "a")
    else:
        output_file = open(arguments.outputFile, "w")
        log_file = open(arguments.outputFile + ".log", "w")
    lstm_model = loadModels(arguments.language)
    if not (arguments.fileFormat).startswith("XML_Large"):
        data_set = loadText(arguments.file, arguments.fileFormat)
        tags = getTags(data_set, lstm_model)
        printResults(generateOutput(data_set, tags, arguments.HTMLFormat), output_file)
        if arguments.evaluate and arguments.fileFormat == "IOB":
            truth = [sentence["format_IOB"] for sentence in data_set]
            precision, recall, f1 = computeMetrics(tags, truth, arguments.fileFormat)
            print(f"Precision: {precision}\nRecall: {recall}\n F1: {f1}")
    elif arguments.fileFormat == "XML_Large_sentences":
        parseAndTagXML(lstm_model, arguments.file, output_file, log_file, False, arguments.lastSentence)
    else:
        parseAndTagXML(lstm_model, arguments.file, output_file, log_file, True, arguments.lastSentence)
    output_file.close()
    log_file.close()


def generateOutput(data_set, tags, html_format):
    if html_format:
        return generateHTML(data_set, tags)
    return annotateText(data_set, tags)


def getTags(data_set, lstm_model):
    addCharAndCasingInformation(data_set)
    testMatrix = createMatrices(data_set, lstm_model.mappings, True)
    return lstm_model.tagSentences(testMatrix)


def loadModels(lang):
    # Which GPU to use for . -1 for CPU
    if torch.cuda.is_available():
        print("Using CUDA")
        bert_cuda_device = 0
    else:
        print("Using CPU")
        bert_cuda_device = -1

    bert_model_name = "bert-base-multilingual-cased"
    fasttext_embeddings = False
    if lang == "en":
        embeddings_file = './embeddings/komninos_english_embeddings.gz'
        model_path = "./models/zagreb/conll2003_ner_0.9250_0.8854_13.h5"
    else:
        embeddings_file = f"./embeddings/fastText157/cc.{lang}.300.vec.gz.top1.bin"
        generic_model_path = f"./models/zagreb/282NER_{lang}_*.h5"
        model_path = glob.glob(generic_model_path)
        if len(model_path) > 1:
            print("Multiple models for the same language.")
            sys.exit(1)
        elif len(model_path) == 0:
            print("Model not found.")
            sys.exit(1)
        model_path = model_path[0]
        fasttext_embeddings = True
    if not os.path.exists(embeddings_file):
        if lang == "en":
            subprocess.run(["sh", "./embeddings/komninos_embeddings.sh"], check=True)
        else:
            subprocess.run(["sh", "./embeddings/fasttext_embeddings.sh", lang], check=True)
    lstm_model = BERTBiLSTM.loadModel(model_path, bert_model_name, bert_cuda_device, embeddings_file, use_fastext=fasttext_embeddings)
    return lstm_model


def loadText(file, file_format):
    if file_format == "tokens":
        data_set = readCoNLL(file, {0: "tokens"})
    elif file_format == "IOB":
        data_set = readCoNLL(file, {1: "tokens", 3: "format_IOB"})
    elif file_format == "XML_tokenized":
        data_set = parseXML(file, True)
    elif file_format == "XML_sentences":
        data_set = parseXML(file, False)
    elif file_format == "plain":
        input_file = open(file, "r", encoding='utf-8')
        data_set = [{'tokens': x.split()} for x in input_file.read().split("\n") if len(x) > 0]
        input_file.close()
    else:
        print("Unsupported format")
        sys.exit(1)
    return data_set


def printResults(results, output_file):
    output_file.write(results)


def annotateText(data_set, tags):
    annotated_text = ""
    model_name = list(tags.keys())[0]
    for sentence_id, sentence in enumerate(data_set):
        annotated_sentence = ""
        tokens = sentence['tokens']
        if len(tokens) != len(tags[model_name][sentence_id]):
            print("Fatal error, tags doesn't match sentences")
            sys.exit(1)
        if "tokens_ids" in sentence:
            annotated_sentence = annotated_sentence.join(f"{counter + 1}\t{token}\t{token_id}\t{tag}\n" for ((counter, (token, token_id)), tag) in zip(enumerate(zip(tokens, sentence["tokens_ids"])), tags[model_name][sentence_id]))
        else:
            annotated_sentence = annotated_sentence.join(f"{counter+1}\t{token}\tX---X\t{tag}\n" for ((counter, token), tag) in zip(enumerate(tokens), tags[model_name][sentence_id]))
        annotated_text += f"{annotated_sentence}\n"
    return annotated_text


def generateHTML(sentences, tags):
    parts = ['<html>' +
             '<head>' +
             '<style>' +
             'hr {border-width: 10px;}' +
             '.button {background-color: #FFFFFF;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer;border-radius: 8px;}' +
             '.ORG {background-color: #4CAF50;}' +
             '.PER {background-color: #008CBA;}' +
             '.LOC {background-color: #f44336;}' +
             '.MISC {background-color: #e7e7e7; color: black;}' +
             '.ISC {background-color: #e7e7e7; color: black;}' +
             '.EVT {background-color: #555555;}' +
             '.PRO {background-color: #555555;}' +
             '.O {color: black;}' +
             '</style>' +
             '</head>' +
             '<body>',
             '</body>' +
             '</html>']

    # :: Output to stdout ::
    lines = ''
    modelName = list(tags.keys())[0]
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']
        for tokenIdx in range(len(tokens)):
            tokenTag = tags[modelName][sentenceIdx][tokenIdx]
            tokenTagPrev = tags[modelName][sentenceIdx][tokenIdx - 1] if tokenIdx > 0 else 'X'
            tokenTagNext = tags[modelName][sentenceIdx][tokenIdx + 1] if tokenIdx < len(tokens) - 1 else 'X'
            tokenTag = str(tokenTag[-4:]) if len(tokenTag) == 6 else str(tokenTag[-3:]) if len(
                tokenTag) > 3 else tokenTag
            tokenTagPrev = str(tokenTagPrev[-3:]) if len(tokenTagPrev) > 3 else tokenTagPrev
            tokenTagNext = str(tokenTagNext[-3:]) if len(tokenTagNext) > 3 else tokenTagNext
            if tokenTag == 'O':
                lines += '<button class="button ' + tokenTag + '">' + tokens[tokenIdx] + '</button>'
            elif tokenTag not in (tokenTagPrev, tokenTagNext):
                lines += '<button class="button ' + tokenTag + '">' + tokens[
                    tokenIdx] + '&nbsp;&nbsp;&nbsp;&nbsp;<small><b>' + tokenTag + '</b></small></button>'
            elif tokenTag != tokenTagPrev and tokenTag == tokenTagNext:
                lines += '<button class="button ' + tokenTag + '">' + tokens[tokenIdx]
            elif tokenTag == tokenTagPrev and tokenTag == tokenTagNext:
                lines += ' ' + tokens[tokenIdx]
            elif tokenTag == tokenTagPrev and tokenTag != tokenTagNext:
                lines += ' ' + tokens[
                    tokenIdx] + '&nbsp;&nbsp;&nbsp;&nbsp;<small><b>' + tokenTag + '</b></small></button>'
            lines += ''
        lines += '<br><br><hr><br><br>'

    return parts[0] + lines + parts[1]


def parseAndTagXML(lstm_model, input_file, output_file, log_file, tokenized, last_sentence_processed):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    xml_parser_tagger = XMLParserTagger(tokenized, lstm_model, output_file, log_file, last_sentence_processed)
    parser.setContentHandler(xml_parser_tagger)
    parser.parse(input_file)


class XMLParserTagger(xml.sax.ContentHandler):
    def __init__(self, tokenized, lstm_model, output_file, log_file, last_sentence_processed):
        self.__tokenized = tokenized
        self.__lstm_model = lstm_model
        self.__current_tag = ""
        self.__current_word = ""
        self.__sentences = []
        self.__sentences_counter = 0
        self.__output_file = output_file
        self.__sentence_id = 0
        self.__last_sentence_processed = last_sentence_processed
        self.__skip_line = False
        self.__log_file = log_file
        if self.__tokenized:
            self.__current_sentence = {"tokens": [], "tokens_ids": []}
        else:
            self.__current_sentence = {}

    def __resetSentences(self):
        self.__sentences = []
        self.__sentences_counter = 0

    def startElement(self, name, attrs):
        self.__current_tag = name
        if name == "s":
            self.__sentence_id = int(attrs["id"])
            if self.__last_sentence_processed > -1:
                if self.__sentence_id <= self.__last_sentence_processed:
                    self.__skip_line = True
                    return
                self.__skip_line = False
            if self.__tokenized:
                self.__current_sentence = {"tokens": [], "tokens_ids": []}
            else:
                self.__current_sentence = {}
        elif name == "w":
            self.__current_word = ""
            if self.__tokenized:
                self.__current_sentence["tokens_ids"].append(attrs["id"])

    def characters(self, content):
        if self.__skip_line:
            return
        content = content.rstrip()
        if len(content) != 0:
            if self.__current_tag == "s":
                if self.__tokenized:
                    print(f"Sentence {self.__sentence_id} isn't tokenized. Applying a simple tokenization based on spaces.\n")
                    self.__log_file.write(f"Sentence {self.__sentence_id} isn't tokenized. Applying a simple tokenization based on spaces.\n")
                    offset = len(self.__current_sentence["tokens"]) + 1
                    for token_id, token in enumerate(content.split()):
                        self.__current_sentence["tokens"].append(token)
                        self.__current_sentence["tokens_ids"].append(f"w{self.__sentence_id}.{token_id+offset}")
                else:
                    if len(self.__current_sentence) == 0:
                        self.__current_sentence = {'tokens': content.split()}
                    else:
                        for token in content.split():
                            self.__current_sentence["tokens"].append(token)
            if self.__current_tag == "w":
                self.__current_word += content

    def endElement(self, name):
        if self.__skip_line:
            return
        if name == "w":
            self.__current_sentence["tokens"].append(self.__current_word)
        elif name == "s":
            self.__sentences.append(self.__current_sentence)
            self.__sentences_counter += 1
            if self.__sentences_counter == 10000:
                self.__callTagger()
                self.__resetSentences()

    def endDocument(self):
        self.__callTagger()

    def __callTagger(self):
        tags = getTags(self.__sentences, self.__lstm_model)
        printResults(annotateText(self.__sentences, tags), self.__output_file)

if __name__ == '__main__':
    main_command()
