import xml.sax

def parseXML(file, tokenized=True):
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # override the default ContextHandler
    xml_parser = XMLParser(tokenized)
    parser.setContentHandler(xml_parser)
    parser.parse(file)
    return xml_parser.getSentences()


class XMLParser(xml.sax.ContentHandler):
    def __init__(self, tokenized):
        self.__currentTag = ""
        self.__currentWord = ""
        self.__sentences = []
        self.__tokenized= tokenized
        if tokenized:
            self.__currentSentence = {"tokens": [], "tokens_ids": []}
        else:
            self.__currentSentence = {}

    def getSentences(self):
        return self.__sentences

    def startElement(self, name, attrs):
        self.__currentTag = ""
        if name == "s":
            if self.__tokenized:
                self.__currentSentence = {"tokens": [], "tokens_ids": []}
            else:
                self.__currentTag = "s"
                self.__currentSentence = {}
        elif name == "w":
            self.__currentTag = "w"
            self.__currentWord = ""
            if self.__tokenized:
                self.__currentSentence["tokens_ids"].append(attrs["id"])

    def characters(self, content):
        if self.__currentTag == "s":
            if len(self.__currentSentence) == 0:
                self.__currentSentence = {'tokens': content.split()}
            else:
                for token in content.split():
                    self.__currentSentence["tokens"].append(token)
        if self.__currentTag == "w":
            self.__currentWord += content

    def endElement(self, name):
        if name == "w":
            self.__currentSentence["tokens"].append(self.__currentWord)
        elif name == "s":
            self.__sentences.append(self.__currentSentence)
