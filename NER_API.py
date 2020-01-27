from flask import Flask
from flask_restplus import Api, Resource, reqparse
from NER import main as ner, processText

NER_API = Flask(__name__)
api = Api(NER_API)


@api.route('/predict/<string:language>')
class NER(Resource):

    @api.doc(params={
        'language': 'Language in which the analysis will be done',
        'text': 'Text to process'
    })
    def post(self, language):
        parser = reqparse.RequestParser()
        parser.add_argument("text")
        args = parser.parse_args()

        if args["text"] != "":
            print(args["text"])
            split_text = [{'tokens': x.split()} for x in args["text"].split("\n") if len(x) > 0]
            split_text = processText(split_text)
            return ner(language, split_text), 200

        return "Text empty", 401


NER_API.run()
