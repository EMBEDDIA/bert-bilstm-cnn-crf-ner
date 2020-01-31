import argparse
import glob
import os
import subprocess
import sys

import torch

from neuralnets.BERTBiLSTM import BERTBiLSTM
from util import ThreeColumnData
from util.DataSet import DataSet
from util.BIOF1Validation import convertIOBtoBIO, computeMetrics
from util.CoNLL import readCoNLL
from util.preprocessing import createMatrices, addCharAndCasingInformation


def main_command():
    arguments_parser = argparse.ArgumentParser()
    required_arguments = arguments_parser.add_argument_group('required arguments')
    required_arguments.add_argument("-f", "--file", help="File to process", required=True)
    required_arguments.add_argument("-l", "--language", help="Language of the file to process (en|hr)", required=True)

    optional_arguments = arguments_parser.add_argument_group('optional arguments')
    optional_arguments.add_argument("-F", "--fileFormat", help="Format of input file [plain, conll, 3cIO, 3cIOB]. Default plain", default="plain")
    optional_arguments.add_argument("-o", "--outputFile", help="Output file", default="")
    optional_arguments.add_argument("-H", "--HTMLFormat", help="Output file as HTML", action="store_true")
    optional_arguments.add_argument("-e", "--evaluate", help="Evaluate (only valid with 3cIO or 3cIOB files)", action="store_true")

    arguments = arguments_parser.parse_args()

    if arguments.outputFile == "":
        if not os.path.exists("./results"):
            os.mkdir("./results")
        arguments.outputFile = f"./results/{arguments.language}.html"

    print(f"File to process: {arguments.file}")
    print(f"Language to process: {arguments.language}")
    print(f"Output redirected to:", {arguments.outputFile})
    data_set = loadText(arguments.file, arguments.fileFormat)
    tags = getTags(arguments.language, data_set.getTokenizedSentences())
    data_set.setPredictedTags(tags)
    printResults(generateOutput(data_set, arguments.HTMLFormat), arguments.outputFile)
    if arguments.evaluate:
        if data_set.hasTrueTags():
            precision, recall, f1 = computeMetrics(data_set.getPreditedTags(), data_set.getTrueTags())
            print(f"Precision: {precision}\nRecall: {recall}\n F1: {f1}")
        else:
            print("The data set isn't annotated with the true tags")


def generateOutput(data_set: DataSet, html_format):
    if html_format:
        return generateHTML(data_set.getTokenizedSentences(), data_set.getPreditedTags())
    return annotateText(data_set.getTokenizedSentences(), data_set.getPreditedTags())


def getTags(language, split_text):
    # Which GPU to use for . -1 for CPU
    if torch.cuda.is_available():
        print("Using CUDA")
        bert_cuda_device = 0
    else:
        print("Using CPU")
        bert_cuda_device = -1
    lstm_model = loadModels(language, bert_cuda_device)
    return predict(split_text, lstm_model)


def loadModels(lang, bert_cuda_device):
    bert_model_name = "bert-base-uncased"
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
    data_set = DataSet()
    if file_format == "conll":
        split_text = readCoNLL(file, {0: "tokens"})
    elif file_format == "3cIOB":
        data_set = ThreeColumnData.processFile(file, data_set)
        split_text = data_set.getTokenizedSentences()
    elif file_format == "3cIO":
        data_set = ThreeColumnData.processFile(file, data_set, ioTags = True)
        split_text = data_set.getTokenizedSentences()
    elif file_format == "plain":
        input_file = open(file, "r", encoding='utf-8')
        split_text = [{'tokens': x.split()} for x in input_file.read().split("\n") if len(x) > 0]
        input_file.close()
    else:
        print("Unsupported format")
        sys.exit(1)
    addCharAndCasingInformation(split_text)
    data_set.setTokenizedSentences(split_text)
    return data_set


def predict(split_text, lstm_model):
    data_matrix = createMatrices(split_text, lstm_model.mappings, True)
    tags = lstm_model.tagSentences(data_matrix)
    model_name = list(tags.keys())[0]
    convertIOBtoBIO(tags[model_name])
    return tags


def printResults(results, output_file):
    file = open(output_file, "w")
    file.write(results)
    file.close()


def annotateText(sentences, tags):
    annotated_text = ""
    model_name = list(tags.keys())[0]
    for sentence_id, sentence in enumerate(sentences):
        annotated_sentence = ""
        tokens = sentence['tokens']
        if len(tokens) != len(tags[model_name][sentence_id]):
            print("Fatal error, tags doesn't match sentences")
            sys.exit(1)
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


if __name__ == '__main__':
    main_command()
