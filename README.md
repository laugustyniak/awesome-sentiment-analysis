# Awesome Sentiment Analysis

A curated list of awesome sentiment analysis frameworks, libraries, 
software (by language), and of course academic papers and methods. In 
addition NLP lib useful in sentiment analysis. Inspired by 
awesome-machine-learning.

If you want to contribute to this list (please do), send me a pull request or
 contact me [@luk_augustyniak](https://twitter.com/luk_augustyniak)
Also, a listed repository should be deprecated if:

* Repository's owner explicitly say that "this library is not maintained".
* Not committed for long time (2~3 years).

## Table of Contents

<!-- MarkdownTOC depth=4 -->

- [Libraries](#lib)
- [Lexicons, Datasets, Word embeddings, and other resources](#data)
- [Tutorials](#tuts)
- [Papers](#papers)
- [Demos](#demos)
- [API](#api)


<!-- /MarkdownTOC -->

<a name="lib" />

## Libraries

* [Python, Textlytics](https://github.com/laugustyniak/textlytics) - set of 
sentiment analysis examples based on Amazon Data, SemEval, IMDB etc.

* [Polish Sentiment Model](https://github.com/riomus/polish-sentiment) - 
Sentiment analysis for polish language using SVM and BoW - within Docker.

* [Python, Spacy](https://spacy.io/) - Industrial-Strength Natural Language 
Processing in Python, one of the best and the fastest libs for NLP. spaCy 
excels at large-scale information extraction tasks. It's written from the 
ground up in carefully memory-managed Cython. Independent research has 
confirmed that spaCy is the fastest in the world. If your application needs 
to process entire web dumps, spaCy is the library you want to be using.

* [Python, TextBlob](https://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers)- TextBlob allows you to specify which algorithms
 you want to use under the hood of its simple API.

* [Python, pattern](http://www.clips.ua.ac.be/pages/pattern-en#sentiment) - 
The pattern.en module contains a fast part-of-speech tagger for English 
(identifies nouns, adjectives, verbs, etc. in a sentence), sentiment 
analysis, tools for English verb conjugation and noun singularization & 
pluralization, and a WordNet interface. 

* [JAVA, CoreNLP by Stanford](http://stanfordnlp.github.io/CoreNLP/) - [Deeply 
Moving: Deep Learning for Sentiment Analysis](http://nlp.stanford.edu/sentiment/).

* [R, TM](http://cran.r-project.org/web/packages/tm/index) - R text mining 
module including tm.plugin.sentiment.

* [Software, GATE](https://gate.ac.uk/sentiment/) - GATE is open source 

* [JAVA, LingPipe](http://alias-i.com/lingpipe/) - LingPipe is tool kit for 

* [Python, NLTK](http://www.nltk.org) - Natural Language Toolkit.

* [Software, KNIME](https://www.knime.org/blog/sentiment-analysis)

* [Software, RapidMiner](https://rapidminer.com/solutions/text-mining/) - 
software capable of solving almost any text processing problem.
processing text using computational linguistics.

* [JAVA, OpenNLP](https://opennlp.apache.org/) - The Apache OpenNLP library is 
a machine learning based toolkit for the processing of natural language text. 

<a name="data"/>

## Lexicons, Datasets, Word embeddings, and other resources

* [AFINN: List of English words rated for valence](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)

* [SentiWordNet: Lexical resource devised for supporting sentiment analysis](http://sentiwordnet.isti.cnr.it/)
[paper](https://www.researchgate.net/publication/220746537_SentiWordNet_30_An_Enhanced_Lexical_Resource_for_Sentiment_Analysis_and_Opinion_Mining)

* [Stanford Sentiment Treebank: Sentiment dataset with fine-grained sentiment 
annotations](http://nlp.stanford.edu/sentiment/code.html) [paper](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

Word Embeddings:
* [GloVe: Algorithm for obtaining word vectors. Pretrained word vectors 
available for download](http://nlp.stanford.edu/projects/glove/) [paper](http://nlp.stanford.edu/pubs/glove.pdf)

* [Word2Vec by Mikolov](https://code.google.com/archive/p/word2vec/) [paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

* [Word2Vec Python lib](https://github.com/RaRe-Technologies/gensim) - Google's
 word2vec reimplementation written in Python (cython). There are also doc2vec 
 and topic modelling method. 

SemEval Challenges - International Workshop on Semantic Evaluation [web](http://aclweb.org/aclwiki/index.php?title=SemEval_3):
* [SemEval2014](http://alt.qcri.org/semeval2014/index.php?id=tasks)

* [SemEval2015](http://alt.qcri.org/semeval2015/index.php?id=tasks)

* [SemEval2016](http://alt.qcri.org/semeval2016/index.php?id=tasks)

* [SemEval2017](http://alt.qcri.org/semeval2017/index.php?id=tasks)

* [SemEval2018](http://alt.qcri.org/semeval2018/) New challenge - 

<a name="tuts" />

## Tutorials

* [SAS2015](https://github.com/laugustyniak/sas2015) iPython Notebook brief 
introduction to Sentiment Analysis in Python @Sentiment Analysis Symposium 2015. Scikit-learn + BoW + SemEval Data.

* [LingPipe Sentiment](http://alias-i.com/lingpipe/demos/tutorial/sentiment/read-me.html)

<a name="papers" />

## Papers

<a name="demos" />

## Demos

* [Sentiment TreeBank](http://nlp.stanford.edu:8080/sentiment/rntnDemo.html)

* [NLTK Demo](http://text-processing.com/demo/sentiment/)

* [GATE Brexit Analyzer](https://cloud.gate.ac.uk/shopfront/displayItem/29)

* [Vivekn's sentiment model](https://github.com/vivekn/sentiment/) and [web 
example](http://sentiment.vivekn.com/)

* [FormTitan](https://prediction.formtitan.com/sentiment-analysis)

<a name="api" />

## API

* [AlchemyAPI - IBM Watson](https://www.ibm.com/watson/developercloud/)

* [NL Google Cloud](https://cloud.google.com/natural-language/)

* [www.sentimentanalysisonline](http://www.sentimentanalysisonline.com)

* [MeaningCloud](https://www.meaningcloud.com/products/sentiment-analysis)

* [texsie](http://texsie.stride.ai)