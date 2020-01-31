from util.DataSet import DataSet
from util.BIOF1Validation import convertIOBtoBIO


def processFile(file, data_set: DataSet, ioTags=False) -> DataSet:
    input_file = open(file, "r", encoding='utf-8')
    sentences = []
    sentence = {"tokens": []}
    sentences_true_tags = []
    sentence_true_tags = []

    for line in input_file:
        if line == "\n":
            if bool(sentence["tokens"]):
                sentences.append(sentence)
                sentence = {"tokens": []}
                sentences_true_tags.append(sentence_true_tags)
                sentence_true_tags = []
            continue
        line_elements = line.split("\t")
        sentence["tokens"].append(line_elements[1])
        sentence_true_tags.append(line_elements[3].rstrip())

    if bool(sentence["tokens"]):
        sentences.append(sentence)
        sentences_true_tags.append(sentence_true_tags)
    input_file.close()

    if ioTags:
        convertIOBtoBIO(sentences_true_tags)

    data_set.setTokenizedSentences(sentences)
    data_set.setTrueTags(sentences_true_tags)

    return data_set
