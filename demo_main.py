#import nltk
import sys
import torch
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation, readCoNLL
from neuralnets.BERTBiLSTM import BERTBiLSTM
from neuralnets.BERTWordEmbeddings import BERTWordEmbeddings

def generateHTML(sentences, tags, outputFile, debug=False):
    parts = ['<html>'+
    '<head>'+
    '<style>'+
    'hr {border-width: 10px;}'+
    '.button {background-color: #FFFFFF;border: none;color: white;padding: 10px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer;border-radius: 8px;}'+
    '.ORG {background-color: #4CAF50;}'+
    '.PER {background-color: #008CBA;}'+
    '.LOC {background-color: #f44336;}'+
    '.MISC {background-color: #e7e7e7; color: black;}'+
    '.ISC {background-color: #e7e7e7; color: black;}'+
    '.EVT {background-color: #555555;}'+
    '.PRO {background-color: #555555;}'+
    '.O {color: black;}'+
    '</style>'+
    '</head>'+
    '<body>',
    '</body>'+
    '</html>']

    # :: Output to stdout ::
    lines = ''
    modelName = list(tags.keys())[0]
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']
        for tokenIdx in range(len(tokens)):
            tokenTag = tags[modelName][sentenceIdx][tokenIdx]
            tokenTagPrev = tags[modelName][sentenceIdx][tokenIdx-1] if tokenIdx > 0 else 'X'
            tokenTagNext = tags[modelName][sentenceIdx][tokenIdx+1] if tokenIdx < len(tokens)-1 else 'X'
            tokenTag = str(tokenTag[-4:]) if len(tokenTag)==6 else str(tokenTag[-3:]) if len(tokenTag)>3 else tokenTag
            tokenTagPrev = str(tokenTagPrev[-3:]) if len(tokenTagPrev)>3 else tokenTagPrev
            tokenTagNext = str(tokenTagNext[-3:]) if len(tokenTagNext)>3 else tokenTagNext
            if tokenTag == 'O':
                lines += '<button class="button '+tokenTag+'">'+tokens[tokenIdx]+'</button>'
            elif tokenTag != tokenTagPrev and tokenTag != tokenTagNext:
                lines += '<button class="button '+tokenTag+'">'+tokens[tokenIdx]+'&nbsp;&nbsp;&nbsp;&nbsp;<small><b>'+tokenTag+'</b></small></button>'
            elif tokenTag != tokenTagPrev and tokenTag == tokenTagNext:
                lines += '<button class="button '+tokenTag+'">'+tokens[tokenIdx]
            elif tokenTag == tokenTagPrev and tokenTag == tokenTagNext:
                lines += ' '+tokens[tokenIdx]
            elif tokenTag == tokenTagPrev and tokenTag != tokenTagNext:
                lines += ' '+tokens[tokenIdx]+'&nbsp;&nbsp;&nbsp;&nbsp;<small><b>'+tokenTag+'</b></small></button>'
            lines += ''
        lines += '<br><br><hr><br><br>'

    if debug:
        print(lines)
    if outputFile != "":
        file = open(outputFile, "w")
        file.write(parts[0]+lines+parts[1])
        file.close()
    else:
        print(parts[0]+lines+parts[1])

#Which GPU to use for . -1 for CPU
if torch.cuda.is_available():
    print("Using CUDA")
    bert_cuda_device = 0
else:
    print("Using CPU")
    bert_cuda_device = -1



modelPath = "models/zagreb/conll2003_ner_0.9250_0.8854_13.h5"
inputPath = "input.conll"
inputColumns = {0: "tokens"}

embeddings_file = 'embeddings/komninos_english_embeddings.gz'
bert_mode = 'weighted_average'



#bert_path_name = '/data6T/Datasets/BERT/cased_L-12_H-768_A-12/'
#Now it is necessary to use this name for accessing to the bert pretrained model
bert_path_name = "bert-base-uncased"

#English

# :: Prepare the input ::
sentences = readCoNLL(inputPath, inputColumns)
print("\n".join([" - "+" ".join(x['tokens']) for x in sentences]))
addCharInformation(sentences)
addCasingInformation(sentences)

# :: Load the model ::
lstmModel = BERTBiLSTM.loadModel(modelPath, bert_path_name, bert_cuda_device, embeddings_file)

# :: Map casing and character information to integer indices ::
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)


# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


generateHTML(sentences, tags, "./results/English.html")

print("English finished.")

#Croatian

modelPath = "models/zagreb/282NER_hr_0.8636_0.8619_24.h5"

# :: embeddings file
embeddings_file = 'embeddings/fastText157/cc.hr.300.vec.gz.top1'

# :: Load the model ::
lstmModel = BERTBiLSTM.loadModel(modelPath, bert_path_name, bert_cuda_device, embeddings_file, use_fastext=True)

inputFile = open("./data/Croatian_example.txt", "r", encoding='utf-8')
text = inputFile.read()
inputFile.close()

sentences = [{'tokens': x.split()} for x in text.split("\n") if len(x) > 0]
addCharInformation(sentences)
addCasingInformation(sentences)

# :: Map casing and character information to integer indices ::
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)


# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)

generateHTML(sentences, tags, "./results/Croatian.html")