# BiLSTM-CNN-CRF with BERT for Sequence Tagging

This repository is based on [BiLSTM-CNN-CRF ELMo implementation](https://github.com/UKPLab/elmo-bilstm-cnn-crf).

The model here present is the one presented in Deliverable 2.2 of Embeddia Project.

The dependencies for running the code are present in the `environement.yml` file. These can be used to create a Anaconda environement.

To run the code, you need to do:

`python NER.py -f {input_file} -F {input_type} -l {language} -o {output_file}`

The argument `-F` can support:
```
plain Plain text, it will tokenize the text based on white spaces
tokens The text is already tokenized and has a format similar to CONLL 2003. Where the first column correspond to the tokens.
IOB The text is already tokenized and has a format similar to CONLL 2003 however, in this case it is already annotated. The second column correspond to the tokens, the third to the labels
XML_tokenized We can support XML files that have the text already split in sentences and tokenized. The format is the one used by http://opus.nlpl.eu/
XML_sentences We can support XML files that have the text already split in sentences. The sentences will be tokenized using white spaces. The format is the one used by http://opus.nlpl.eu/
XML_Large_tokenized Similar to XML_tokenized, however, we store the last sentence processed in order to recover if needed.
XML_Large_sentences Similar to XML_sentences, however, we store the last sentence processed in order to recover if needed.
```

It has some optional arguments, such as:

```
--HTMLFormat -H This makes the output to be an HTML file that contains the original text annotated with the predicted labels
--evaluate -e If the document is annotated, it evaluates the document
--lastSentence -s This argument is used when large XML files are parsed and we need to restart the process of it. It takes the id of the last processed sentence.
```

# License

This work has been attributed with an MIT License, with an exception to the code that it was developed by a third-party. Specifically, we have used code from:

- [BiLSTM-CNN-CRF ELMo implementation](https://github.com/UKPLab/elmo-bilstm-cnn-crf): Apache 2.0 License
- [phipleg/keras](https://github.com/phipleg/keras/tree/crf): MIT License

All the files have their respective license.
