from flask import Flask
from flask_restplus import Api, Resource, reqparse
from NER import getTags, generateOutput
from util.DataSet import DataSet
from util.preprocessing import addCharAndCasingInformation

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
            addCharAndCasingInformation(split_text)
            data_set = DataSet()
            data_set.setTokenizedSentences(split_text)
            tags = getTags(language, data_set.getTokenizedSentences())
            data_set.setPredictedTags(tags)
            return generateOutput(data_set, False), 200

        return "Text empty", 401


NER_API.run()
