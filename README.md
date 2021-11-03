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

## How to cite?

```
@inproceedings{moreno-etal-2019-tlr,
    title = "{TLR} at {BSNLP}2019: A Multilingual Named Entity Recognition System",
    author = "Moreno, Jose G.  and
      Linhares Pontes, Elvys  and
      Coustaty, Mickael  and
      Doucet, Antoine",
    booktitle = "Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-3711",
    doi = "10.18653/v1/W19-3711",
    pages = "83--88"
}

```

# Parent project

This work is is result of the European Union H2020 Project [Embeddia](http://embeddia.eu/). Embeddia is a project that creates NLP tools that focuses on European under-represented languages and that has for objective to improve the accessibility of these tools to the general public and to media enterprises. Visit [Embeddia's Github](https://github.com/orgs/EMBEDDIA/) to discover more NLP tools and models created within this project.

