import argparse
import os
import subprocess

import torch

from neuralnets.BERTBiLSTM import BERTBiLSTM
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation


def main():
    arguments_parser = argparse.ArgumentParser()
    required_arguments = arguments_parser.add_argument_group('required arguments')
    required_arguments.add_argument("-f", "--file", help="File to process", required=True)
    required_arguments.add_argument("-l", "--language", help="Language of the file to process", required=True)

    optional_arguments = arguments_parser.add_argument_group('optional arguments')
    optional_arguments.add_argument("-s", "--split", help="File needs to be split.", action="store_true")
    optional_arguments.add_argument("-o", "--outputFile", help="Output file", default="")

    arguments = arguments_parser.parse_args()
    if arguments.outputFile == "":
        if not os.path.exists("./results"):
            os.mkdir("./results")
        arguments.outputFile = f"./results/{arguments.language}.html"
    print(f"File to process: {arguments.file}")
    print(f"Language to process: {arguments.language}")
    # Which GPU to use for . -1 for CPU
    if torch.cuda.is_available():
        print("Using CUDA")
        bert_cuda_device = 0
    else:
        print("Using CPU")
        bert_cuda_device = -1
    lstm_model = loadModels(arguments.language, bert_cuda_device)
    split_text = loadText(arguments.file, arguments.split)
    predict(split_text, lstm_model, arguments.outputFile)


def loadModels(lang, bert_cuda_device):
    model_path = "./models/zagreb/conll2003_ner_0.9250_0.8854_13.h5"
    bert_model_name = "bert-base-uncased"
    fasttext_embeddings = False
    if lang == "en":
        embeddings_file = './embeddings/komninos_english_embeddings.gz'
    else:
        embeddings_file = f"./embeddings/fastText157/cc.{lang}.300.vec.gz.top1"
        fasttext_embeddings = True
    if not os.path.exists(embeddings_file):
        if lang == "en":
            subprocess.run(["sh", "./embeddings/komninos_embeddings.sh"])
        else:
            subprocess.run(["sh", "./embeddings/fasttext_embeddings.sh", lang])
    lstm_model = BERTBiLSTM.loadModel(model_path, bert_model_name, bert_cuda_device, embeddings_file, use_fastext=fasttext_embeddings)
    return lstm_model


def loadText(file, split):
    input_file = open(file, "r", encoding='utf-8')
    if not split:
        split_text = input_file.read()
    else:
        split_text = [{'tokens': x.split()} for x in input_file.read().split("\n") if len(x) > 0]
    input_file.close()
    addCharInformation(split_text)
    addCasingInformation(split_text)
    return split_text


def predict(split_text, lstm_model, output_file):
    data_matrix = createMatrices(split_text, lstm_model.mappings, True)
    tags = lstm_model.tagSentences(data_matrix)
    printResults(generateHTML(split_text, tags), output_file)


def printResults(results, output_file):
    # folder = re.sub(r"[^/]+/$", "", output_file)
    file = open(output_file, "w")
    file.write(results)
    file.close()


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
            elif tokenTag != tokenTagPrev and tokenTag != tokenTagNext:
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
    main()
