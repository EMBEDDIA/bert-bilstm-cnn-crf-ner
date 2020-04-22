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
