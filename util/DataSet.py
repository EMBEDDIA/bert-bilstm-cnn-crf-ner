class DataSet:
    def __init__(self):
        self.__tokenized_sentences = []
        self.__predicted_tags = []
        self.__true_tags = []
        self.__has_true_tags = False

    def setTokenizedSentences(self, tokenized_sentences):
        self.__tokenized_sentences = tokenized_sentences

    def getTokenizedSentences(self):
        return self.__tokenized_sentences

    def setPredictedTags(self, predicted_tags):
        self.__predicted_tags = predicted_tags

    def getPreditedTags(self):
        return self.__predicted_tags

    def setTrueTags(self, true_tags):
        self.__true_tags = true_tags
        if bool(true_tags):
            self.__has_true_tags = True

    def getTrueTags(self):
        return self.__true_tags

    def hasTrueTags(self):
        return self.__has_true_tags