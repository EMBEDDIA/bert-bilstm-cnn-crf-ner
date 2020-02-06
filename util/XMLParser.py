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
            self.__currentSentence = {"tokens": []}
        else:
            self.__currentSentence = {}

    def getSentences(self):
        return self.__sentences

    def startElement(self, tag, attributes):
        self.__currentTag = ""
        if tag == "s":
            if self.__tokenized:
                self.__currentSentence = {"tokens": []}
            else:
                self.__currentTag = "s"
                self.__currentSentence = {}
        elif tag == "w":
            self.__currentTag = "w"
            self.__currentWord = ""

    def characters(self, content):
        if self.__currentTag == "s":
            if len(self.__currentSentence) == 0:
                self.__currentSentence = {'tokens': content.split()}
            else:
                for token in content.split():
                    self.__currentSentence["tokens"].append(token)
        if self.__currentTag == "w":
            self.__currentWord += content

    def endElement(self, tag):
        if tag == "w":
            self.__currentSentence["tokens"].append(self.__currentWord)
        elif tag == "s":
            self.__sentences.append(self.__currentSentence)
