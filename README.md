# Awesome Sentiment Analysis

A curated list of awesome sentiment analysis frameworks, libraries, software (by language), and of course academic papers and methods. In addition NLP lib useful in sentiment analysis. Inspired by awesome-machine-learning.

If you want to contribute to this list (please do), send me a pull request or contact me [@luk_augustyniak](https://twitter.com/luk_augustyniak)

## Table of Contents

<!-- MarkdownTOC depth=4 -->

- [Libraries](#libraries)
    - [Aspect-Based Sentiment Analysis](#aspect-based-sentiment-analysis)
- [Resources](#resources)
    - [Lexicons](#lexicons)
    - [Datasets](#datasets)
    - [Word Embeddings](#word-embeddings)
    - [Language Models](#language-models)
- [International Workshops](#international-workshops)
- [Papers](#papers)
    - [Language Models](#language-models)
    - [Neural Network based Models](#neural-network-based-models)
    - [Lexicon-based Ensembles](#lexicon-based-ensembles)
- [Tutorials](#tutorials)
- [Books](#books)
- [Demos](#demos)
- [API](#api)
- [Related Studies](#related-studies)

<!-- /MarkdownTOC -->

## Libraries

* [Python, Textlytics](https://github.com/laugustyniak/textlytics) - set of sentiment analysis examples based on Amazon Data, SemEval, IMDB etc.

* [Java, Polish Sentiment Model](https://github.com/riomus/polish-sentiment) - Sentiment analysis for polish language using SVM and BoW - within Docker.

* [Python, Spacy](https://spacy.io/) - Industrial-Strength Natural Language Processing in Python, one of the best and the fastest libs for NLP. spaCy excels at large-scale information extraction tasks. It's written from the ground up in carefully memory-managed Cython. Independent research has confirmed that spaCy is the fastest in the world. If your application needs to process entire web dumps, spaCy is the library you want to be using.

* [Python, TextBlob](https://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers) - TextBlob allows you to specify which algorithms you want to use under the hood of its simple API.

* [Python, pattern](http://www.clips.ua.ac.be/pages/pattern-en#sentiment) - The pattern.en module contains a fast part-of-speech tagger for English (identifies nouns, adjectives, verbs, etc. in a sentence), sentiment analysis, tools for English verb conjugation and noun singularization & pluralization, and a WordNet interface.

* [Java, CoreNLP by Stanford](http://stanfordnlp.github.io/CoreNLP/) - NLP toolkit with [Deeply Moving: Deep Learning for Sentiment Analysis](http://nlp.stanford.edu/sentiment/).

* [R, TM](http://cran.r-project.org/web/packages/tm/index) - R text mining module including tm.plugin.sentiment.

* [Software, GATE](https://gate.ac.uk/sentiment/) - GATE is open source software capable of solving almost any text processing problem.

* [Java, LingPipe](http://alias-i.com/lingpipe/) - LingPipe is tool kit for processing text using computational linguistics.

* [Python, NLTK](http://www.nltk.org) - Natural Language Toolkit.

* [C++, MITIE](https://github.com/mit-nlp/MITIE) - MIT Information Extraction.

* [Software, KNIME](https://www.knime.org/blog/sentiment-analysis) - KNIME® Analytics Platform is the leading open solution for data-driven innovation, helping you discover the potential hidden in your data, mine for fresh insights, or predict new futures. Our enterprise-grade, open source platform is fast to deploy, easy to scale and intuitive to learn. With more than 1000 modules, hundreds of ready-to-run examples, a comprehensive range of integrated tools, and the widest choice of advanced algorithms available, KNIME Analytics Platform is the perfect toolbox for any data scientist. Our steady course on unrestricted open source is your passport to a global community of data scientists, their expertise, and their active contributions.

* [Software, RapidMiner](https://rapidminer.com/solutions/text-mining/) - software capable of solving almost any text processing problem. processing text using computational linguistics.

* [JAVA, OpenNLP](https://opennlp.apache.org/) - The Apache OpenNLP library is a machine learning based toolkit for the processing of natural language text.

* [Dragon Sentiment Classifier C#](https://github.com/amrish7/Dragon) - Dragon Sentiment API is a C# implementation of the Naive Bayes Sentiment Classifier to analyze the sentiment of a text corpus.

* [sentiment: Tools for Sentiment Analysis in R](https://github.com/timjurka/sentiment) - sentiment is an R package with tools for sentiment analysis including bayesian classifiers for positivity/negativity and emotion classification.

* [ASUM Java](http://uilab.kaist.ac.kr/research/WSDM11/) - Aspect and Sentiment Unification Model for Online Review Analysis.

* [AFINN-based sentiment analysis for Node.js](https://github.com/thisandagain/sentiment) - Sentiment is a Node.js module that uses the AFINN-165 wordlist and Emoji Sentiment Ranking to perform sentiment analysis on arbitrary blocks of input text.

* [SentiMental - Putting the Mental in Sentimental in js](https://github.com/thinkroth/Sentimental) - Sentiment analysis tool for node.js based on the AFINN-111 wordlist. Version 1.0 introduces performance improvements making it both the first, and now fastest, AFINN backed Sentiment Analysis tool for node.

[Back to Top](#table-of-contents)

### Aspect-based Sentiment Analysis

* [Twitter-sent-dnn](https://github.com/xiaohan2012/twitter-sent-dnn) - Deep Neural Network for Sentiment Analysis on Twitter.

* [Aspect Based Sentiment Analysis](https://github.com/pedrobalage/SemevalAspectBasedSentimentAnalysis) - System that participated in Semeval 2014 task 4: Aspect Based Sentiment Analysis.

* [Aspect Based Sentiment Analysis using End-to-End Memory Networks](https://github.com/ganeshjawahar/mem_absa) - TensorFlow implementation of [Tang et al.'s EMNLP 2016 work](https://arxiv.org/abs/1605.08900).

* [Generating Reviews and Discovering Sentiment](https://github.com/openai/generating-reviews-discovering-sentiment) - Code for [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444) (Alec Radford, Rafal Jozefowicz, Ilya Sutskever).

* [Sentiment Analysis with Social Attention](https://github.com/yiyang-gt/social-attention) - Code for the TACL paper [Overcoming Language Variation in Sentiment Analysis with Social Attention](https://arxiv.org/abs/1511.06052)

* [Neural Sentiment Classification](https://github.com/thunlp/NSC) - Neural Sentiment Classification aims to classify the sentiment in a document with neural models, which has been the state-of-the-art methods for sentiment classification. In this project, we provide our implementations of NSC, NSC+LA and NSC+UPA [Chen et al., 2016] in which user and product information is considered via attentions over different semantic levels.

[Back to Top](#table-of-contents)

## Resources

### Lexicons

* [Multidomain Sentiment Lexicons](https://github.com/laugustyniak/textlytics/tree/master/textlytics/data/lexicons) - lexicons from 10 domains based on Amazon Product Dataset extracted using method described in 
    [paper](https://www.cs.rpi.edu/~szymansk/papers/C3-ASONAM14.pdf) and used in 
    [paper](http://www.mdpi.com/1099-4300/18/1/4).

* [AFINN](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010) - AFINN is a list of English words rated for valence with an integer between minus five (negative) and plus five (positive). The words have been manually labeled by Finn Årup Nielsen in 2009-2011.

* [SentiWordNet](http://sentiwordnet.isti.cnr.it/) 
    [[paper]](https://www.researchgate.net/publication/220746537_SentiWordNet_30_An_Enhanced_Lexical_Resource_for_Sentiment_Analysis_and_Opinion_Mining) - Lexical resource based on WordNet

* [SentiWords](https://hlt-nlp.fbk.eu/technologies/sentiwords) - Collection of 155,000 English words with a sentiment score included between -1 and 1. Words are in the form lemma#PoS and are aligned with WordNet lists that include adjectives, nouns, verbs and adverbs.

* [SenticNet](https://sentic.net/) 
    [[API]](http://sentic.net/api/) - Words with a sentiment score included between -1 and 1.

* [WordStat](https://provalisresearch.com/products/content-analysis-software/wordstat-dictionary/sentiment-dictionaries/) - Context-specific sentiment analysis dictionary with categories Negative, Positive, Uncertainty, Litigiousness and Modal. This dataset is inspired from two papers, written by Loughran and McDonald (2011) and Young and Soroka (2011).

* [MPQA (Multi-Perspective Question Answering) Subjectivity Lexicon](http://mpqa.cs.pitt.edu/lexicons/) - The MPQA (Multi-Perspective Question Answering) Subjectivity Lexicon is a list of subjectivity clues that is part of [OpinionFinder](http://mpqa.cs.pitt.edu/opinionfinder/opinionfinder_2/) and also helps to determine text polarity.

* [NRC-Canada Lexicons](http://saifmohammad.com/WebPages/lexicons.html) - the web page lists various word association lexicons that capture word-sentiment, word-emotion, and word-colour associations. 

* [Sentiment140](http://saifmohammad.com/Lexicons/Sentiment140-Lexicon-v0.1.zip) - One of the NRC-Canada team lexicon - the Sentiment140 Lexicon is a list of words and their associations with positive and negative sentiment. The lexicon is provides sentiment score for unigrams, bigrams and unigram-bigram pairs.

* [MSOL](http://saifmohammad.com/Lexicons/MSOL-June15-09.txt.zip) - Macquarie Semantic Orientation Lexicon.

* [SemEval-2015 English Twitter Sentiment Lexicon](http://saifmohammad.com/WebDocs/lexiconstoreleaseonsclpage/SemEval2015-English-Twitter-Lexicon.zip) - The lexicon was used as an official test set in the [SemEval-2015 shared Task #10: Subtask E](http://alt.qcri.org/semeval2015/task10/). The phrases in this lexicon include at least one of these [negators](http://saifmohammad.com/WebDocs/lexiconstoreleaseonsclpage/SemEval2015-English-negators.txt). 

* [SemEval-2016 Arabic Twitter Sentiment Lexicon](http://saifmohammad.com/WebDocs/lexiconstoreleaseonsclpage/SemEval2016-Arabic-Twitter-Lexicon.zip) - The lexicon was used as an official test set in the [SemEval-2016 shared Task #7: Detecting Sentiment Intensity of English and Arabic Phrases](http://alt.qcri.org/semeval2016/task7/). The phrases in this lexicon include at least one of these [negators](http://saifmohammad.com/WebDocs/list-Arabic-negators.txt).    

* [SemEval-2016 English Twitter Mixed Polarity Lexicon](http://saifmohammad.com/WebDocs/lexiconstoreleaseonsclpage/SCL-OPP.zip) - This SCL, referred to as the Sentiment Composition Lexicon of Opposing Polarity Phrases (SCL-OPP), includes phrases that have at least one positive and at least one negative word—for example, phrases such as happy accident, best winter break, couldn’t stop smiling, and lazy sundays. We refer to such phrases as opposing polarity phrases. SCL-OPP has 265 trigrams, 311 bigrams, and 602 unigrams annotated with real-valued sentiment association scores through Best-Worst scaling (aka MaxDiff).

* [SemEval-2016 General English Sentiment Modifiers Lexicon](http://saifmohammad.com/WebDocs/lexiconstoreleaseonsclpage/SCL-NMA.zip) - Sentiment Composition Lexicon of Negators, Modals, and Adverbs (SCL-NMA). Negators, modals, and degree adverbs can significantly affect the sentiment of the words they modify. We manually annotate a set of phrases that include negators (such as no and cannot), modals (such as would have been and could), degree adverbs (such as quite and less), and their combinations. Both the phrases and their constituent content words are annotated with real-valued scores of sentiment intensity using the technique Best–Worst Scaling (aka MaxDiff), which provides reliable annotations. We refer to the resulting lexicon as Sentiment Composition Lexicon of Negators, Modals, and Adverbs (SCL-NMA). The lexicon was used as an official test set in the [SemEval-2016 shared Task #7: Detecting Sentiment Intensity of English and Arabic Phrases](http://alt.qcri.org/semeval2016/task7/). The objective of that task was to automatically predict sentiment intensity scores for multi-word phrases.

* [The NRC Valence, Arousal, and Dominance Lexicon](http://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon-Aug2018Release.zip) - The NRC Valence, Arousal, and Dominance (VAD) Lexicon includes a list of more than 20,000 English words and their valence, arousal, and dominance scores. For a given word and a dimension (V/A/D), the scores range from 0 (lowest V/A/D) to 1 (highest V/A/D). The lexicon with its fine-grained real-valued scores was created by manual annotation using Best--Worst Scaling. 

* [EmoLex NRC Word-Emotion Association Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) - the NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). The annotations were manually done by crowdsourcing.

* [WN-Affect emotion Lexicon](http://wndomains.fbk.eu/wnaffect.html) - WordNet-Affect is an extension of WordNet Domains, including a subset of synsets suitable to represent affective concepts correlated with affective words. Similarly to our method for domain labels, we assigned to a number of WordNet synsets one or more affective labels (a-labels). In particular, the affective concepts representing emotional state are individuated by synsets marked with the a-label emotion. There are also other a-labels for those concepts representing moods, situations eliciting emotions, or emotional responses.

* [EmoLex NRC Word-Emotion Association Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) - the NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). The annotations were manually done by crowdsourcing.

* [Multidimensional Stance Lexicon](https://github.com/umashanthi-research/multidimensional-stance-lexicon) - A Multidimensional Lexicon for Interpersonal Stancetaking. Pavalanathan, Fitzpatrick, Kiesling, and Eisenstein. ACL 2017.

* [WN-Affect emotion Lexicon](http://wndomains.fbk.eu/wnaffect.html) - WordNet-Affect is an extension of WordNet Domains, including a subset of synsets suitable to represent affective concepts correlated with affective words. Similarly to our method for domain labels, we assigned to a number of WordNet synsets one or more affective labels (a-labels). In particular, the affective concepts representing emotional state are individuated by synsets marked with the a-label emotion. There are also other a-labels for those concepts representing moods, situations eliciting emotions, or emotional responses.

[Back to Top](#table-of-contents)

### Datasets

* [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/code.html) 
    [[paper]](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) - Sentiment dataset with fine-grained sentiment annotations. The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis, originally collected by [Pang and Lee](https://arxiv.org/abs/cs/0506075). In their work on sentiment treebanks, Socher et al. used Amazon's Mechanical Turk to create fine-grained labels for all parsed phrases in the corpus. This competition presents a chance to benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. You are asked to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

* [Amazon Product Dataset](http://jmcauley.ucsd.edu/data/amazon/) - This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). The updated version of dataset - update as for 2018 is availalbe here [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html). 
 
* [IMDB Movies Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) - This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. Authors provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing.

* [Sentiment Labelled Sentences Dataset](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) The dataset contains sentences labelled with positive or negative sentiment. This dataset was created for the following 
[paper](http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf). It contains sentences labelled with positive or negative sentiment. Score is either 1 (for positive) or 0 (for negative) The sentences come from three different websites/fields: imdb.com, amazon .com, yelp.com. For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.

* [sentic.net](http://sentic.net/) -  concept-level sentiment analysis, that is, performing tasks such as polarity detection and emotion recognition by leveraging on semantics and linguistics in stead of solely relying on word co-occurrence frequencies.

[Back to Top](#table-of-contents)

### Word Embeddings

* [WordNet2Vec](https://arxiv.org/pdf/1606.03335.pdf) - Corpora Agnostic Word Vectorization Method based on WordNet.

* [GloVe](http://nlp.stanford.edu/projects/glove/) 
    [[paper]](http://nlp.stanford.edu/pubs/glove.pdf) - Algorithm for obtaining word vectors. Pretrained word vectors available for download.

* [Word2Vec by Mikolov](https://code.google.com/archive/p/word2vec/) 
    [[paper]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - Google's original code and pretrained word embeddings.

* [Word2Vec Python lib](https://github.com/RaRe-Technologies/gensim) - Google's word2vec reimplementation written in Python (cython). There are also doc2vec and topic modelling method.

[Back to Top](#table-of-contents)

### Pretrained Language Models

* BERT (Encoder of the transormer) 
    * [Tensorflow-based ](https://github.com/google-research/bert) Implementation: 
        * BERT<sub>base</sub>,
        BERT<sub>large</sub>
        BERT<sub>multilingual</sub>, etc.
    * [Torch-based  (Higging Face)](https://huggingface.co/models) model implementations:
        * XLNet, XmlRoBERTa, etc.
* GPT (Decoder of the transformer)
    * [GPT-2](https://huggingface.co/gpt2)

### International Workshops

* SemEval Challenges International Workshop on Semantic Evaluation 
    [[site]](http://aclweb.org/aclwiki/index.php?title=SemEval_3)
* SemEval 
    [[2014]](http://alt.qcri.org/semeval2014/index.php?id=tasks)
    [[2015]](http://alt.qcri.org/semeval2015/index.php?id=tasks)
    [[2016]](http://alt.qcri.org/semeval2016/index.php?id=tasks)
    [[2017]](http://alt.qcri.org/semeval2017/index.php?id=tasks)
    [[2018]](http://alt.qcri.org/semeval2018/) -- New challenge for 2018 year, waiting for confirmation about tasks etc.

[Back to Top](#table-of-contents)

## Papers

### Language Models

* [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) -- 
is a generalized autoregressive pretraining method that (1) enables
learning bidirectional contexts by maximizing the expected likelihood over all
permutations of the factorization order and (2) overcomes the limitations of BERT
thanks to its autoregressive formulation

* [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) --
authors conduct exhaustive experiments to investigate different fine-tuning methods of 
[BERT](https://arxiv.org/pdf/1810.04805.pdf) 
(Bidirectional Encoder Representations from Transformers) on text
classification task and provide a general solution for BERT fine-tuning

### Neural Network based Models

* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) -  convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks.

### Lexicon-based Ensembles 

* [Comprehensive Study on Lexicon-based Ensemble Classification Sentiment Analysis](http://www.mdpi.com/1099-4300/18/1/4) - Comparison of several lexicon, supervised learning and ensemble methods for sentiment analysis.

* [Simpler is better? Lexicon-based ensemble sentiment classification beats supervised methods](https://www.cs.rpi.edu/~szymansk/papers/C3-ASONAM14.pdf) - lexicon-based ensemble can beat supervised learning.

[Back to Top](#table-of-contents)

## Tutorials

* [GPT2 For Text Classification using Hugging Face Transformers](https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/) - GPT model application for sentiment analysis task

* [SAS2015](https://github.com/laugustyniak/sas2015) iPython Notebook brief introduction to Sentiment Analysis in Python @ Sentiment Analysis Symposium 2015. Scikit-learn + BoW + SemEval Data.

* [LingPipe Sentiment](http://alias-i.com/lingpipe/demos/tutorial/sentiment/read-me.html) - This tutorial covers assigning sentiment to movie reviews using language models. There are many other approaches to sentiment. One we use fairly often is sentence based sentiment with a logistic regression classifier. Contact us if you need more information. For movie reviews we focus on two types of classification problem: Subjective (opinion) vs. Objective (fact) sentences Positive (favorable) vs. Negative (unfavorable) movie reviews

* [Stanford's cs224d lectures on Deep Learning for Natural Language Processing](https://cs224d.stanford.edu/lectures/) - course provided by Richard Socher.

[Back to Top](#table-of-contents)

## Books

* [Sentiment Analysis: mining sentiments, opinions, and emotions](https://www.cs.uic.edu/~liub/FBS/sentiment-opinion-emotion-analysis.html) - This book is suitable for students, researchers, and practitioners interested in natural language processing in general, and sentiment analysis, opinion mining, emotion analysis, debate analysis, and intention mining in specific. Lecturers can use the book in class.

[Back to Top](#table-of-contents)

## Demos

* [Sentiment TreeBank](http://nlp.stanford.edu:8080/sentiment/rntnDemo.html) - demo of Stanford's Treebank Sentiment Analysis
* [NLTK Demo](http://text-processing.com/demo/sentiment/)
* [GATE Brexit Analyzer](https://cloud.gate.ac.uk/shopfront/displayItem/29)
* [Vivekn's sentiment model](https://github.com/vivekn/sentiment/) and [web example](http://sentiment.vivekn.com/)
* [FormTitan](https://prediction.formtitan.com/sentiment-analysis)

[Back to Top](#table-of-contents)

## API

* [AlchemyAPI - IBM Watson](https://www.ibm.com/watson/developercloud/)
* [www.sentimentanalysisonline](http://www.sentimentanalysisonline.com)
* [MeaningCloud](https://www.meaningcloud.com/products/sentiment-analysis)
* [texsie](http://texsie.stride.ai)
* [Google Cloud API](https://cloud.google.com/natural-language/docs/sentiment-tutorial)
* [Microsoft Azure Text Analytics API](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
* [Aylien](https://developer.aylien.com/text-api-demo)
* [Amazon Comprehend](https://aws.amazon.com/comprehend/features/)
* [MS Cognitive Services](https://azure.microsoft.com/en-gb/services/cognitive-services/text-analytics/)

[Back to Top](#table-of-contents)

## Related Studies

* [Benchmarking Multimodal Sentiment Analysis](http://sentic.net/benchmarking-multimodal-sentiment-analysis.pdf)  - multimodal sentiment analysis and emotion detection (text, audio and video).

[Back to Top](#table-of-contents)
