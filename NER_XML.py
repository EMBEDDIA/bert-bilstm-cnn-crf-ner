import xml.sax


def main(file):

    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # override the default ContextHandler
    parser.setContentHandler(XMLParser())
    parser.parse(file)


class XMLParser(xml.sax.ContentHandler):
    def __init__(self):
        self.currentTag = ""
        self.currentSentence = []
        self.currentWord = ""

    def startElement(self, tag, attributes):
        self.currentTag = ""
        if tag == "s":
            self.currentSentence = []
        elif tag == "w":
            self.currentTag = "w"
            self.currentWord = ""

    def characters(self, content):
        if self.currentTag == "w":
            self.currentWord += content

    def endElement(self, tag):
        if tag == "w":
            self.currentSentence.append(self.currentWord)
        elif tag == "s":
            print(self.currentSentence)


if __name__ == '__main__':
    main("/home/adrian/Desktop/example_es.xml")
