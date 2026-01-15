# Awesome Sentiment Analysis

A curated list of awesome sentiment analysis frameworks, libraries, software (by language), and of course academic papers and methods. In addition NLP lib useful in sentiment analysis. Inspired by awesome-machine-learning.

**Latest Update (January 2026)**: Comprehensive update covering 2021-2026 advances including:
- Large Language Models (GPT-4, Claude, Llama, Gemini, Mixtral)
- Modern Transformers (RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa)
- Multimodal Sentiment Analysis (vision-language models, multimodal LLMs)
- Multilingual and Cross-lingual Methods
- Recent Benchmarks and Datasets (2023-2026)
- Domain-Specific Applications (Financial, Healthcare, Social Media)

If you want to contribute to this list (please do), send me a pull request or contact me [@luk_augustyniak](https://twitter.com/luk_augustyniak)

## Table of Contents

<!-- MarkdownTOC depth=4 -->

- [Libraries](#libraries)
    - [Modern Transformer-based Libraries (2023-2026)](#modern-transformer-based-libraries-2023-2026)
    - [Traditional Libraries](#traditional-libraries)
    - [Aspect-Based Sentiment Analysis](#aspect-based-sentiment-analysis)
- [Resources](#resources)
    - [Lexicons](#lexicons)
    - [Datasets](#datasets)
        - [Classic Benchmarks](#classic-benchmarks)
        - [Recent Datasets (2023-2026)](#recent-datasets-2023-2026)
        - [Domain-Specific Datasets](#domain-specific-datasets)
    - [Word Embeddings](#word-embeddings)
    - [Pretrained Language Models](#pretrained-language-models)
        - [Large Language Models (2023-2026)](#large-language-models-2023-2026)
        - [Encoder-based Transformers (BERT Family)](#encoder-based-transformers-bert-family)
        - [Multilingual Transformers](#multilingual-transformers)
        - [Domain-Specific Models](#domain-specific-models)
        - [Decoder-based Models](#decoder-based-models)
        - [Hybrid Architectures (2023-2025)](#hybrid-architectures-2023-2025)
- [Multimodal Sentiment Analysis](#multimodal-sentiment-analysis)
    - [Overview](#overview)
    - [Recent Models and Frameworks (2024-2025)](#recent-models-and-frameworks-2024-2025)
    - [Multimodal LLMs for Sentiment Analysis](#multimodal-llms-for-sentiment-analysis)
    - [Key Research Findings (2024-2025)](#key-research-findings-2024-2025)
    - [Applications](#applications)
- [Multilingual and Cross-lingual Sentiment Analysis](#multilingual-and-cross-lingual-sentiment-analysis)
    - [State-of-the-Art Models (2024-2025)](#state-of-the-art-models-2024-2025)
    - [Recent Approaches and Techniques](#recent-approaches-and-techniques)
    - [Performance Benchmarks](#performance-benchmarks)
    - [Supported Languages](#supported-languages)
- [International Workshops](#international-workshops)
- [Papers](#papers)
    - [Language Models](#language-models)
    - [Transformer Models and RoBERTa (2023-2025)](#transformer-models-and-roberta-2023-2025)
    - [Multimodal Sentiment Analysis (2024-2025)](#multimodal-sentiment-analysis-2024-2025)
    - [Multilingual and Cross-lingual Sentiment Analysis (2024-2025)](#multilingual-and-cross-lingual-sentiment-analysis-2024-2025)
    - [Aspect-Based Sentiment Analysis (2024-2025)](#aspect-based-sentiment-analysis-2024-2025)
    - [Domain-Specific Applications (2024-2025)](#domain-specific-applications-2024-2025)
    - [Neural Network based Models](#neural-network-based-models)
    - [Lexicon-based Ensembles](#lexicon-based-ensembles)
- [Tutorials](#tutorials)
- [Books](#books)
- [Demos](#demos)
- [API](#api)
- [Related Studies](#related-studies)

<!-- /MarkdownTOC -->

## Libraries

### Modern Transformer-based Libraries (2023-2026)

* [Python, Hugging Face Transformers](https://huggingface.co/transformers) - State-of-the-art Natural Language Processing library with 215+ sentiment analysis models. Supports BERT, RoBERTa, DistilBERT, ALBERT, XLNet, and all modern transformer architectures. Simple 5-line integration for sentiment analysis with pre-trained models.

* [Python, cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) - RoBERTa model fine-tuned for Twitter sentiment analysis, achieving state-of-the-art performance on social media text (updated 2024).

* [Python, ModernFinBERT](https://huggingface.co/ModernFinBERT) - Financial sentiment analysis model based on ModernBERT architecture (released January 2025), specialized for financial texts, earnings calls, and analyst reports.

* [Python, tabularisai/multilingual-sentiment-analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) - Multilingual sentiment model supporting multiple languages simultaneously (released December 2024).

* [Python, Flair](https://github.com/flairNLP/flair) - Modern NLP framework with multilingual support and state-of-the-art sentiment analysis models, particularly strong for cross-lingual tasks.

* [Python, Stanza](https://stanfordnlp.github.io/stanza/) - Stanford NLP library with multilingual support for 60+ languages, includes sentiment analysis capabilities.

### Traditional Libraries

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

#### Classic Benchmarks

* [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/code.html)
    [[paper]](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) - Sentiment dataset with fine-grained sentiment annotations. The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis, originally collected by [Pang and Lee](https://arxiv.org/abs/cs/0506075). In their work on sentiment treebanks, Socher et al. used Amazon's Mechanical Turk to create fine-grained labels for all parsed phrases in the corpus. This competition presents a chance to benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. You are asked to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

* [Amazon Product Dataset](http://jmcauley.ucsd.edu/data/amazon/) - This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). The updated version of dataset - update as for 2018 is availalbe here [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html).

* [IMDB Movies Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) - This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. Authors provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing.

* [Sentiment Labelled Sentences Dataset](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) The dataset contains sentences labelled with positive or negative sentiment. This dataset was created for the following
[paper](http://mdenil.com/media/papers/2015-deep-multi-instance-learning.pdf). It contains sentences labelled with positive or negative sentiment. Score is either 1 (for positive) or 0 (for negative) The sentences come from three different websites/fields: imdb.com, amazon .com, yelp.com. For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.

* [sentic.net](http://sentic.net/) -  concept-level sentiment analysis, that is, performing tasks such as polarity detection and emotion recognition by leveraging on semantics and linguistics in stead of solely relying on word co-occurrence frequencies.

#### Recent Datasets (2023-2026)

* [**Brand24/MMS - Massively Multilingual Sentiment Corpus**](https://huggingface.co/datasets/Brand24/mms)
    [[arXiv]](https://arxiv.org/abs/2306.07902)
    [[NeurIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html)
    [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2023/file/7945ab41f2aada1247a7c95e75cdf6c8-Paper-Datasets_and_Benchmarks.pdf)
    [[github]](https://github.com/Brand24-AI/mms_benchmark)
    [[benchmark]](https://huggingface.co/spaces/Brand24/mms_benchmark) - **The most extensive open massively multilingual corpus** for training sentiment models. Accepted to **NeurIPS 2023 Datasets and Benchmarks Track**. Contains 79 manually selected high-quality datasets from over 350 sources covering **27 languages** across 6 language families with **6,164,762 training samples**. Features rich linguistic metadata including morphological, syntactic, and functional properties, plus data quality confidence scores. Presents multi-faceted sentiment classification benchmark with hundreds of experiments on different base models, training objectives, and fine-tuning strategies. Languages include: Arabic, Bulgarian, Chinese, Czech, Dutch, English, Spanish, French, Japanese, Polish, Portuguese, Russian, and 15 others. Class distribution: Positive (56.7%), Neutral (21.8%), Negative (21.6%). License: CC BY-NC 4.0.

* [TweetEval](https://huggingface.co/datasets/tweet_eval) - Part of ACL initiative for semantic evaluation. Widely used benchmark for Twitter sentiment analysis and text classification tasks (2020-2025).

* [TweetFinSent](https://github.com/TweetFinSent) - Financial sentiment dataset from Twitter. State-of-the-art models achieve 69.54% accuracy and 65.72% macro F1-score with adversarial training (2023-2024).

* [IMDB Deep Context Reviews](https://www.kaggle.com/datasets) - Extended version capturing movie reviews with richer contextual information from IMDB's vast user base (2024-2025).

* [Large-scale English Comment Dataset](https://huggingface.co/datasets) - Collection of 241,000+ English-language comments from various online platforms (updated 2025).

* [MLDoc Dataset](https://github.com/facebookresearch/MLDoc) - Multilingual document classification corpus used for cross-lingual sentiment analysis. State-of-the-art adversarial training achieves 88.48% average accuracy (2024).

* [PAWS-X](https://github.com/google-research-datasets/paws) - Paraphrase Adversaries from Word Scrambling, cross-lingual dataset achieving 86.63% accuracy with recent methods (2024).

* [Kurdish Medical Corpus](https://github.com/Kurdish-BLARK) - Specialized medical sentiment dataset achieving 92% accuracy with multilingual BERT (2024).

#### Domain-Specific Datasets

* **Financial Sentiment**
    * [Financial PhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) - Sentences from financial news categorized by sentiment
    * TweetFinSent - Twitter financial sentiment with 69.54% SOTA accuracy (2023)

* **Healthcare/Mental Health**
    * Mental Health sentiment datasets for student wellbeing analysis (2024-2025)
    * Clinical sentiment corpora for patient feedback analysis

* **Restaurant Reviews**
    * Multilingual restaurant review datasets achieving 91.9% accuracy with XLM-RSA (2024)

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

#### Large Language Models (2023-2026)

* **GPT Family (OpenAI)**
    * [GPT-4](https://openai.com/gpt-4) - Advanced large language model with strong sentiment analysis capabilities, particularly for complex emotional nuances and context-dependent sentiment (2023-2024)
    * [GPT-4o](https://openai.com/index/hello-gpt-4o/) - Multimodal version with enhanced performance (2024)
    * [GPT-3.5 Turbo](https://platform.openai.com/docs/models) - Cost-effective alternative for sentiment analysis tasks

* **Claude Family (Anthropic)**
    * [Claude 4.5](https://www.anthropic.com/claude) - Achieved 75% average accuracy across sentiment analysis tasks, with 82% accuracy in emotion detection (2025 benchmark)
    * Claude 3.5 Sonnet - High-performance model for nuanced sentiment understanding

* **Llama Family (Meta)**
    * [Llama 3.1](https://github.com/meta-llama/llama-models) - Open-source LLM with strong sentiment analysis performance in multilingual contexts (2024)
    * [Llama 2](https://llama.meta.com/) - Widely used for fine-tuning on domain-specific sentiment tasks (2023)

* **Gemini (Google)**
    * [Gemini Pro](https://ai.google.dev/gemini-api) - Multimodal LLM with sentiment analysis capabilities across text and images (2024-2025)

* **Mixtral (Mistral AI)**
    * [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) - Mixture-of-experts model showing competitive performance in sentiment classification (2024)

* **Grok (xAI)**
    * [Grok 4](https://grok.x.ai/) - Specialized in monitoring online sentiment and identifying emerging trends on social media (2024-2025)

#### Encoder-based Transformers (BERT Family)

* **BERT (Bidirectional Encoder Representations from Transformers)**
    * [BERT-base, BERT-large](https://github.com/google-research/bert) - Original TensorFlow implementation (Google, 2018)
    * [BERT Multilingual](https://huggingface.co/bert-base-multilingual-cased) - Supports 104 languages
    * Typical performance: 87.8% accuracy on sentiment tasks

* **RoBERTa (Robustly Optimized BERT)**
    * [RoBERTa-base, RoBERTa-large](https://huggingface.co/roberta-base) - Improved BERT training achieving 88.5-96.30% accuracy (Facebook AI, 2019)
    * [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) - Fine-tuned for social media (69.54% on TweetFinSent)
    * Consistently outperforms BERT on sentiment benchmarks with F1-scores up to 98.11%

* **DistilBERT**
    * [DistilBERT](https://huggingface.co/distilbert-base-uncased) - 40% smaller, 60% faster than BERT while retaining 97% of performance (Hugging Face, 2019)

* **ALBERT (A Lite BERT)**
    * [ALBERT](https://huggingface.co/albert-base-v2) - Parameter-efficient version of BERT with reduced memory consumption (Google, 2019)

* **XLNet**
    * [XLNet](https://huggingface.co/xlnet-base-cased) - Generalized autoregressive pretraining outperforming BERT on several benchmarks (Google/CMU, 2019)

#### Multilingual Transformers

* **XLM-RoBERTa**
    * [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) - Trained on 100 languages, achieves 91.9% accuracy on multilingual sentiment tasks (Facebook AI, 2020)
    * Outperforms other cross-lingual approaches by 3%+ in zero-shot settings

* **mBERT (Multilingual BERT)**
    * [mBERT](https://huggingface.co/bert-base-multilingual-cased) - Supports cross-lingual sentiment analysis with 92% accuracy on specialized corpora

#### Domain-Specific Models

* **Financial Sentiment**
    * [FinBERT](https://huggingface.co/ProsusAI/finbert) - BERT fine-tuned on financial texts (ProsusAI)
    * [ModernFinBERT](https://huggingface.co/ModernFinBERT) - Latest financial sentiment model (2025)
    * [BloombergGPT](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) - 50B parameter LLM for financial NLP including sentiment analysis

* **Healthcare/Mental Health**
    * [RoBERTa-Mental-Health](https://huggingface.co/mental-health) - Specialized models for analyzing student mental health and clinical sentiment

#### Decoder-based Models

* **GPT Family**
    * [GPT-2](https://huggingface.co/gpt2) - Decoder-based transformer (OpenAI, 2019)
    * [GPT-Neo, GPT-J](https://huggingface.co/EleutherAI) - Open-source GPT alternatives (EleutherAI, 2021)

#### Hybrid Architectures (2023-2025)

* **BERT-LSTM Hybrid** - Combining BERT contextual embeddings with BiLSTM for improved sequence dependencies
* **RoBERTa-GRU** - Hybrid models combining transformers with recurrent networks achieving 96.77% accuracy
* **BERT-Attention** - Multi-layered attention mechanisms with BERT for comprehensive sentiment dissection

[Back to Top](#table-of-contents)

## Multimodal Sentiment Analysis

Multimodal sentiment analysis combines text, images, video, and audio to understand sentiment more comprehensively than text-only approaches.

### Overview

* Multimodal Aspect-based Sentiment Analysis (MABSA) has become a core NLP task as user-generated content increasingly includes multiple modalities (text, images, video) (2024-2025)
* Vision-language models demonstrate remarkable potential by integrating visual and textual information to enhance sentiment classification accuracy
* Critical challenges include capturing key information across modalities, achieving cross-modal alignment, and narrowing the semantic gap between image and text

### Recent Models and Frameworks (2024-2025)

* **Sentiment Analysis Engine (SAE)** - End-to-end multimodal model addressing challenges in capturing emotional changes across modalities
    [[paper]](https://dl.acm.org/doi/10.1145/3708359.3712106)

* **RoBERTa-AOBERT Multi-modal Model** - Combines RoBERTa with aspect-oriented BERT for image-text sentiment analysis
    [[paper]](https://dl.acm.org/doi/10.1145/3745533.3745597)

* **Multimodal GRU with Directed Pairwise Cross-Modal Attention** - Advanced architecture for cross-modal sentiment understanding
    [[paper]](https://www.nature.com/articles/s41598-025-93023-3)

* **FDR-MSA (Feature Disentanglement and Reconstruction)** - Novel approach to multimodal sentiment analysis through feature separation and reconstruction
    [[paper]](https://www.sciencedirect.com/science/article/pii/S0950705124005999)

* **Image-Text Sentiment Analysis with Multi-Channel Multi-Modal Joint Learning** - Advanced fusion techniques for analyzing sentiment across image-text pairs
    [[paper]](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2371712)

### Multimodal LLMs for Sentiment Analysis

* **LLaVA (Large Language and Vision Assistant)** - Demonstrates strong capabilities in multimodal aspect-based sentiment analysis
* **GPT-4V (Vision)** - Multimodal GPT-4 variant for analyzing sentiment in images and text
* **Gemini Pro** - Google's multimodal LLM with sentiment analysis across modalities

### Key Research Findings (2024-2025)

* Survey: "Large language models meet text-centric multimodal sentiment analysis" - comprehensive review of LLM applications to multimodal SA
    [[paper]](https://link.springer.com/article/10.1007/s11432-024-4593-8)
* Uncertainty exists about LLM adaptability to multimodal aspect-based sentiment analysis (MABSA), though recent advances show promise
* Multimodal models with multi-layer feature fusion and multi-task learning achieve state-of-the-art results
    [[paper]](https://www.nature.com/articles/s41598-025-85859-6)

### Applications

* Social media sentiment analysis (Twitter, Instagram, TikTok)
* Video content sentiment detection
* Customer feedback analysis with images and text
* Product review analysis combining text and product images

[Back to Top](#table-of-contents)

## Multilingual and Cross-lingual Sentiment Analysis

Analysis of sentiment across multiple languages and transfer of sentiment models between languages.

### State-of-the-Art Models (2024-2025)

* **XLM-RoBERTa (XLM-R)** - Large multilingual Transformer consistently outperforms other cross-lingual approaches in zero-shot classification by 3%+
    [[Hugging Face]](https://huggingface.co/xlm-roberta-base)
    * Trained on 100 languages
    * Achieves 91.9% accuracy on multilingual sentiment tasks

* **XLM-RSA** - Novel multilingual model based on XLM-RoBERTa with Aspect-Focused Attention
    * 91.9% accuracy on restaurant reviews (2024)
    * Surpasses BERT (87.8%) and RoBERTa (88.5%)

* **Multilingual BERT (mBERT)** - Supports 104 languages
    * 92% accuracy on specialized corpora (e.g., Kurdish Medical Corpus)
    * Effective for cross-lingual embedding with MUSE, BiCVM, BiSkip

### Recent Approaches and Techniques

* **Ensemble Methods** - Combining transformers and LLMs for cross-lingual sentiment by translating to base language (English)
    [[paper]](https://www.nature.com/articles/s41598-024-60210-7)

* **Prompt-based Fine-tuning** - Language-independent sentiment analysis using prompt engineering with multilingual transformers
    [[paper]](https://www.nature.com/articles/s41598-025-03559-7)

* **Adaptive Self-alignment** - Bridging resource gaps with data augmentation and transfer learning
    [[paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192753/)

* **Zero-shot and Few-shot Learning** - Small Multilingual Language Models (SMLMs) show superior zero-shot performance vs LLMs; LLMs demonstrate enhanced adaptive potential in few-shot settings (2024)

### Performance Benchmarks

* **Brand24/MMS Benchmark**: The most extensive multilingual benchmark with 79 datasets, 27 languages, 6.16M samples (NeurIPS 2023)
    [[dataset]](https://huggingface.co/datasets/Brand24/mms)
    [[interactive benchmark]](https://huggingface.co/spaces/Brand24/mms_benchmark)
    [[arXiv]](https://arxiv.org/abs/2306.07902)
    [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html)
* **MLDoc Dataset**: 88.48% average accuracy with adversarial training (2024)
* **PAWS-X Dataset**: 86.63% accuracy for cross-lingual paraphrase detection (2024)
* **Restaurant Reviews**: 91.9% with XLM-RSA across multiple languages (2024)

### Supported Languages

Recent models support extensive language coverage including:
* **Brand24/MMS covers 27 languages**: Arabic, Bulgarian, Chinese, Czech, Dutch, English, Spanish, French, Japanese, Polish, Portuguese, Russian, and 15 others across 6 language families
* Major languages: English, Chinese, Spanish, Arabic, French, German, Italian, Portuguese, Russian, Japanese
* Specialized models: Hindi, Korean, Turkish, Kurdish, Polish, and 90+ additional languages

### Applications

* Social media monitoring across global markets
* Customer sentiment analysis for international brands
* Multilingual chatbot sentiment understanding
* Cross-border e-commerce review analysis

[Back to Top](#table-of-contents)

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

* [Sentiment Analysis in the Era of Large Language Models: A Reality Check](https://arxiv.org/pdf/2305.15005.pdf) --
authors evaluate performance across 13 tasks on 26 datasets and compare the large language models (LLMs) such as ChatGPT 
with the results against small language models (SLMs) trained on domain-specific datasets, 
and highlight the limitations of current evaluation practices in assessing LLMs’ SA abilities.

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

### Transformer Models and RoBERTa (2023-2025)

* [Improving sentiment classification using a RoBERTa-based hybrid model](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1292010/full) - Hybrid RoBERTa-GRU model achieving superior performance on sentiment classification (2023)

* [Advancing Sentiment Analysis: Evaluating RoBERTa against Traditional and Deep Learning Models](https://etasr.com/index.php/ETASR/article/view/9703) - Comprehensive comparison showing RoBERTa achieving 96.30% accuracy and F1-scores of 98.11% (2024)

* [Exploring transformer models for sentiment classification: A comparison of BERT, RoBERTa, ALBERT, DistilBERT, and XLNet](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13701) - Comparative study of transformer models with RoBERTa consistently outperforming others (2024)

* [A BERT–LSTM–Attention Framework for Robust Multi-Class Sentiment Analysis on Twitter Data](https://www.mdpi.com/2079-8954/13/11/964) - Hybrid architecture combining BERT with BiLSTM and attention mechanisms for Twitter sentiment (2024)

* [Emotion-Aware RoBERTa enhanced with emotion-specific attention and TF-IDF gating for fine-grained emotion recognition](https://www.nature.com/articles/s41598-025-99515-6) - Enhanced RoBERTa achieving 96.77% accuracy and weighted F1-score of 0.97 (2025)

* [Generalizing sentiment analysis: a review of progress, challenges, and emerging directions](https://link.springer.com/article/10.1007/s13278-025-01461-8) - Comprehensive review covering advances from traditional ML to Transformers and hybrid architectures (2025)

### Multimodal Sentiment Analysis (2024-2025)

* [Large language models meet text-centric multimodal sentiment analysis: a survey](https://link.springer.com/article/10.1007/s11432-024-4593-8) - Comprehensive survey on applying LLMs to multimodal sentiment analysis (2024)

* [Whether Current Large Language Models is Suitable for Multimodal Aspect-based Sentiment Analysis?](https://dl.acm.org/doi/10.1145/3712623.3712644) - Investigation of LLM adaptability to MABSA tasks including Llama2, LLaVA, and ChatGPT (2024)

* [Multimodal sentiment analysis based on multi-layer feature fusion and multi-task learning](https://www.nature.com/articles/s41598-025-85859-6) - Novel approach using multi-layer feature fusion for multimodal SA (2025)

* [FDR-MSA: Enhancing multimodal sentiment analysis through feature disentanglement and reconstruction](https://www.sciencedirect.com/science/article/pii/S0950705124005999) - Advanced feature processing for multimodal sentiment (2024)

* [SAE: A Multimodal Sentiment Analysis Large Language Model](https://dl.acm.org/doi/10.1145/3708359.3712106) - End-to-end LLM for multimodal sentiment analysis (2025)

### Multilingual and Cross-lingual Sentiment Analysis (2024-2025)

* [**Massively Multilingual Corpus of Sentiment Datasets and Multi-faceted Sentiment Classification Benchmark**](https://arxiv.org/abs/2306.07902) - Łukasz Augustyniak, Szymon Woźniak, Marcin Gruza, Piotr Gramacki, Krzysztof Rajda, Mikołaj Morzy, Tomasz Kajdanowicz. **NeurIPS 2023 Datasets and Benchmarks Track**. [[NeurIPS proceedings]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html) [[PDF]](https://proceedings.neurips.cc/paper_files/paper/2023/file/7945ab41f2aada1247a7c95e75cdf6c8-Paper-Datasets_and_Benchmarks.pdf) - **The most extensive open massively multilingual corpus** with 79 high-quality datasets covering 27 languages (6 language families) and 6.16M training samples. Presents multi-faceted sentiment classification benchmark summarizing hundreds of experiments on different base models, training objectives, dataset collections, and fine-tuning strategies. Addresses challenges in multilingual sentiment analysis with rich linguistic metadata including morphological, syntactic, and functional properties. Available on [HuggingFace](https://huggingface.co/datasets/Brand24/mms) with interactive [benchmark](https://huggingface.co/spaces/Brand24/mms_benchmark). [GitHub](https://github.com/Brand24-AI/mms_benchmark)

* [A multimodal approach to cross-lingual sentiment analysis with ensemble of transformer and LLM](https://www.nature.com/articles/s41598-024-60210-7) - Ensemble method combining transformers and LLMs for cross-lingual sentiment (2024)

* [Prompt-based fine-tuning with multilingual transformers for language-independent sentiment analysis](https://www.nature.com/articles/s41598-025-03559-7) - Novel prompt-based approach for multilingual sentiment (2025)

* [Bridging resource gaps in cross-lingual sentiment analysis: adaptive self-alignment with data augmentation and transfer learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192753/) - Addressing resource constraints in cross-lingual SA (2024)

* [Multilingual sentiment analysis in restaurant reviews using aspect focused learning](https://www.nature.com/articles/s41598-025-12464-y) - XLM-RSA achieving 91.9% accuracy on multilingual restaurant reviews (2025)

* [The Model Arena for Cross-lingual Sentiment Analysis: A Comparative Study in the Era of Large Language Models](https://aclanthology.org/2024.wassa-1.12/) - Comparative study of LLMs vs SMLMs in cross-lingual settings (2024)

### Aspect-Based Sentiment Analysis (2024-2025)

* [A systematic review of aspect-based sentiment analysis: domains, methods, and trends](https://link.springer.com/article/10.1007/s10462-024-10906-z) - Comprehensive systematic review of ABSA methods and trends (2024)

* [Large-Scale Aspect-Based Sentiment Analysis with Reasoning-Infused LLMs](https://arxiv.org/html/2601.03940) - Incorporating reasoning techniques into LLMs for ABSA (2025)

* [Aspect-based Sentiment Analysis via Synthetic Image Generation](https://aclanthology.org/2025.findings-emnlp.1190/) - Novel approach generating sentimental images for ABSA (EMNLP 2025)

* [Unifying aspect-based sentiment analysis BERT and multi-layered graph convolutional networks for comprehensive sentiment dissection](https://www.nature.com/articles/s41598-024-61886-7) - Multi-layered Enhanced Graph Convolutional Networks (MLEGCN) for ABSA (2024)

* [Triple dimensional psychology knowledge encouraging graph attention networks to exploit aspect-based sentiment analysis](https://www.nature.com/articles/s41598-025-08914-2) - Psychology-informed graph attention networks (VADGAT) for ABSA (2025)

* [Local interpretation of deep learning models for Aspect-Based Sentiment Analysis](https://www.sciencedirect.com/science/article/pii/S0952197624021067) - Addressing interpretability in deep learning ABSA models (2025)

### Domain-Specific Applications (2024-2025)

* [An overview of model uncertainty and variability in LLM-based sentiment analysis: challenges, mitigation strategies, and the role of explainability](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/full) - LLM challenges in specialized domains (finance, healthcare, legal) (2025)

* [Analyzing student mental health with RoBERTa-Large: a sentiment analysis and data analytics approach](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1615788/full) - Healthcare application for mental health monitoring (2025)

* [Comparative analysis of transformer models for sentiment classification of UK CBDC discourse on X](https://link.springer.com/article/10.1007/s44257-025-00035-4) - Financial sentiment analysis on social media (2025)

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
