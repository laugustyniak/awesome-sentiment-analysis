# Awesome Sentiment Analysis

A curated list of awesome sentiment analysis frameworks, libraries, 
software (by language), and of course academic papers and methods. In 
addition NLP lib useful in sentiment analysis. Inspired by 
awesome-machine-learning.

If you want to contribute to this list (please do), send me a pull request or
 contact me [@luk_augustyniak](https://twitter.com/luk_augustyniak)

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

* [Java, Polish Sentiment Model](https://github.com/riomus/polish-sentiment) - 
Sentiment analysis for polish language using SVM and BoW - within Docker.

* [Python, Spacy](https://spacy.io/) - Industrial-Strength Natural Language 
Processing in Python, one of the best and the fastest libs for NLP. spaCy 
excels at large-scale information extraction tasks. It's written from the 
ground up in carefully memory-managed Cython. Independent research has 
confirmed that spaCy is the fastest in the world. If your application needs 
to process entire web dumps, spaCy is the library you want to be using.

* [Python, TextBlob](https://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers) - TextBlob allows you to specify which algorithms
 you want to use under the hood of its simple API.

* [Python, pattern](http://www.clips.ua.ac.be/pages/pattern-en#sentiment) - 
The pattern.en module contains a fast part-of-speech tagger for English 
(identifies nouns, adjectives, verbs, etc. in a sentence), sentiment 
analysis, tools for English verb conjugation and noun singularization & 
pluralization, and a WordNet interface. 

* [Java, CoreNLP by Stanford](http://stanfordnlp.github.io/CoreNLP/) - 
NLP toolkit with [Deeply Moving: Deep Learning for Sentiment Analysis](http://nlp.stanford.edu/sentiment/).

* [R, TM](http://cran.r-project.org/web/packages/tm/index) - R text mining 
module including tm.plugin.sentiment.

* [Software, GATE](https://gate.ac.uk/sentiment/) - GATE is open source 

* [Java, LingPipe](http://alias-i.com/lingpipe/) - LingPipe is tool kit for 

* [Python, NLTK](http://www.nltk.org) - Natural Language Toolkit.

* [C++, MITIE](https://github.com/mit-nlp/MITIE) - MIT Information Extraction

* [Software, KNIME](https://www.knime.org/blog/sentiment-analysis) - KNIME® 
Analytics Platform is the leading open solution for data-driven innovation, helping you discover the potential hidden in your data, mine for fresh insights, or predict new futures. Our enterprise-grade, open source platform is fast to deploy, easy to scale and intuitive to learn. With more than 1000 modules, hundreds of ready-to-run examples, a comprehensive range of integrated tools, and the widest choice of advanced algorithms available, KNIME Analytics Platform is the perfect toolbox for any data scientist. Our steady course on unrestricted open source is your passport to a global community of data scientists, their expertise, and their active contributions.

* [Software, RapidMiner](https://rapidminer.com/solutions/text-mining/) - 
software capable of solving almost any text processing problem.
processing text using computational linguistics.

* [JAVA, OpenNLP](https://opennlp.apache.org/) - The Apache OpenNLP library is 
a machine learning based toolkit for the processing of natural language text. 

<a name="data"/>

## Lexicons, Datasets, Word embeddings, and other resources

Lexicons:
* [Multidomain Sentiment Lexicons](https://github.com/laugustyniak/textlytics/tree/master/textlytics/data/lexicons) - lexicons
 from 10 domains based on Amazon Product Dataset extracted using method 
 described in [paper](https://www.cs.rpi.edu/~szymansk/papers/C3-ASONAM14.pdf) and used in [paper](http://www.mdpi.com/1099-4300/18/1/4).

* [AFINN](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010) - List of English words rated for valence

* [SentiWordNet](http://sentiwordnet.isti.cnr.it/)
[paper](https://www.researchgate.net/publication/220746537_SentiWordNet_30_An_Enhanced_Lexical_Resource_for_Sentiment_Analysis_and_Opinion_Mining) - Lexical resource based on WordNet

Datasets:
* [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/code.html) 
[paper](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) - Sentiment dataset with fine-grained sentiment 
annotations. The Rotten Tomatoes movie review dataset is a corpus of movie 
reviews used for sentiment analysis, originally collected by [Pang and Lee](https://arxiv.org/abs/cs/0506075). In their work on sentiment treebanks, 
Socher et al. used Amazon's Mechanical Turk to create fine-grained labels
 for all parsed phrases in the corpus. This competition presents a chance to
  benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. 
  You are asked to label phrases on a scale of five values: negative, 
  somewhat negative, neutral, somewhat positive, positive. Obstacles like 
  sentence negation, sarcasm, terseness, language ambiguity, and many others
   make this task very challenging.

* [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/) - This 
dataset contains product reviews and metadata from Amazon, including 142.8 
million reviews spanning May 1996 - July 2014. This dataset includes reviews
 (ratings, text, helpfulness votes), product metadata (descriptions, 
 category information, price, brand, and image features), and links (also 
 viewed/also bought graphs).
 
* [IMDB movies reviews dataset](http://ai.stanford.edu/~amaas/data/sentiment/) - This is a dataset for binary sentiment 
classification containing substantially more data than previous benchmark 
datasets. Authors provide a set of 25,000 highly polar movie reviews for 
training, and 25,000 for testing.

* [Sentiment Labelled Sentences Data Set](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) The dataset contains sentences
 labelled with positive or negative sentiment. This dataset was created for 
 the Paper [From Group to Individual Labels using Deep Features, Kotzias et.
  al,. KDD 2015](http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf). It contains sentences labelled with positive or negative 
  sentiment. Score is either 1 (for positive) or 0 (for negative)	
The sentences come from three different websites/fields: imdb.com, amazon
.com, yelp.com. For each website, there exist 500 positive and 500 negative 
sentences. Those were selected randomly for larger datasets of reviews.  
We attempted to select sentences that have a clearly positive or negative 
connotaton, the goal was for no neutral sentences to be selected.  

* [sentic.net](http://sentic.net/) -  concept-level sentiment analysis, that 
is, performing tasks such as polarity detection and emotion recognition by 
leveraging on semantics and linguistics in stead of solely relying on word 
co-occurrence frequencies.

Word Embeddings:
* [WordNet2Vec](https://arxiv.org/pdf/1606.03335.pdf) - Corpora Agnostic Word Vectorization Method based on WordNet.

* [GloVe](http://nlp.stanford.edu/projects/glove/) [paper](http://nlp.stanford.edu/pubs/glove.pdf) - Algorithm for obtaining word vectors. Pretrained word vectors 
available for download

* [Word2Vec by Mikolov](https://code.google.com/archive/p/word2vec/) [paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - Google's original code and 
pretrained word embeddings. 

* [Word2Vec Python lib](https://github.com/RaRe-Technologies/gensim) - Google's
 word2vec reimplementation written in Python (cython). There are also doc2vec 
 and topic modelling method. 

SemEval Challenges - International Workshop on Semantic Evaluation [web](http://aclweb.org/aclwiki/index.php?title=SemEval_3):
* [SemEval2014](http://alt.qcri.org/semeval2014/index.php?id=tasks)

* [SemEval2015](http://alt.qcri.org/semeval2015/index.php?id=tasks)

* [SemEval2016](http://alt.qcri.org/semeval2016/index.php?id=tasks)

* [SemEval2017](http://alt.qcri.org/semeval2017/index.php?id=tasks)

* [SemEval2018](http://alt.qcri.org/semeval2018/) New challenge for 2018 
year, waiting for confirmation about tasks etc.

* [WN-Affect emotion lexicon](http://wndomains.fbk.eu/wnaffect.html) - WordNet-Affect is an extension of WordNet Domains, including a subset of synsets suitable to represent affective concepts correlated with affective words. Similarly to our method for domain labels, we assigned to a number of WordNet synsets one or more affective labels (a-labels). In particular, the affective concepts representing emotional state are individuated by synsets marked with the a-label emotion. There are also other a-labels for those concepts representing moods, situations eliciting emotions, or emotional responses.

* [EmoLex NRC Word-Emotion Association Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) - the NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). The annotations were manually done by crowdsourcing.

<a name="tuts" />

## Tutorials

* [SAS2015](https://github.com/laugustyniak/sas2015) iPython Notebook brief 
introduction to Sentiment Analysis in Python @ Sentiment Analysis Symposium 
2015. Scikit-learn + BoW + SemEval Data.

* [LingPipe Sentiment](http://alias-i.com/lingpipe/demos/tutorial/sentiment/read-me.html) - This tutorial covers 
assigning sentiment to movie reviews using language models. There are many 
other approaches to sentiment. One we use fairly often is sentence based 
sentiment with a logistic regression classifier. Contact us if you need more 
information. For movie reviews we focus on two types of classification problem:
Subjective (opinion) vs. Objective (fact) sentences Positive (favorable) vs. Negative (unfavorable) movie reviews

* [Stanford's cs224d lectures on Deep Learning for Natural Language 
Processing](https://cs224d.stanford.edu/lectures/) - course provided by 
Richard Socher. 

<a name="papers" />

## Papers, books

* [Comprehensive Study on Lexicon-based Ensemble Classification Sentiment 
Analysis](http://www.mdpi.com/1099-4300/18/1/4) - Comparison of several 
lexicon, supervised learning and ensemble methods for sentiment analysis. 

* [Simpler is better? Lexicon-based ensemble sentiment classification beats 
supervised methods](https://www.cs.rpi.edu/~szymansk/papers/C3-ASONAM14.pdf) - lexicon-based ensemble can beat supervised learning.

* [Sentiment Analysis: mining sentiments, opinions, and emotions](https://www.cs.uic.edu/~liub/FBS/sentiment-opinion-emotion-analysis.html) - This book is
 suitable for students, researchers, and practitioners interested in natural language processing in general, and sentiment analysis, opinion mining, emotion analysis, debate analysis, and intention mining in specific. Lecturers can use the book in class.
  
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) -  convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks.

Multimodal sentiment analysis:
* [Benchmarking Multimodal Sentiment Analysis](http://sentic.net/benchmarking-multimodal-sentiment-analysis.pdf) - multimodal sentiment 
analysis and emotion detection (text, audio and video). 

<a name="demos" />

## Demos

* [Sentiment TreeBank](http://nlp.stanford.edu:8080/sentiment/rntnDemo.html) 
- demo of Stanford's Treebank Sentiment Analysis

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