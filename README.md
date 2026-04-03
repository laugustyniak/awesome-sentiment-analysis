# Awesome Sentiment Analysis

A curated list of awesome sentiment analysis frameworks, libraries, software (by language), and of course academic papers and methods. In addition NLP lib useful in sentiment analysis. Inspired by awesome-machine-learning.

**Latest Update (April 2026)**: Comprehensive update covering 2021-2026 advances including:
- Large Language Models (GPT-4, Claude, Llama, Gemini, Mixtral, DeepSeek)
- Modern Transformers (RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, ModernBERT)
- Multimodal Sentiment Analysis (vision-language models)
- Multilingual and Cross-lingual Methods (Brand24/MMS — NeurIPS 2023, SemEval-2026)
- **NEW: LLM Techniques** — Prompt Engineering, CoT, RAG, LoRA/QLoRA, RLHF, DPO
- **NEW: LLM Evaluation & Benchmarks** — SentiEval, stability metrics, model leaderboard
- **NEW: Explainable Sentiment Analysis** — SHAP, LIME, ModernBERT-XAI, attention viz
- **NEW: LLM Reliability & Safety** — Hallucination, bias, uncertainty quantification
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
- [LLM Techniques for Sentiment Analysis](#llm-techniques-for-sentiment-analysis)
    - [Prompt Engineering](#prompt-engineering)
    - [In-Context Learning & Few-Shot Methods](#in-context-learning--few-shot-methods)
    - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
    - [Instruction Tuning & Alignment](#instruction-tuning--alignment)
- [LLM Evaluation & Benchmarks for Sentiment Analysis](#llm-evaluation--benchmarks-for-sentiment-analysis)
    - [Benchmark Frameworks](#benchmark-frameworks)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Model Performance Leaderboard (2025-2026)](#model-performance-leaderboard-2025-2026)
- [Explainable Sentiment Analysis](#explainable-sentiment-analysis)
    - [Methods & Tools](#methods--tools)
    - [Survey Papers](#survey-papers)
- [LLM Reliability & Safety in Sentiment Analysis](#llm-reliability--safety-in-sentiment-analysis)
    - [Hallucination](#hallucination)
    - [Bias & Fairness](#bias--fairness)
    - [Uncertainty & Variability](#uncertainty--variability)
    - [Domain Instability](#domain-instability)
- [International Workshops](#international-workshops)
- [Papers](#papers)
    - [Language Models](#language-models)
    - [Prompt Engineering & LLM Methods (2025-2026)](#prompt-engineering--llm-methods-2025-2026)
    - [Parameter-Efficient Fine-Tuning (2025-2026)](#parameter-efficient-fine-tuning-2025-2026)
    - [Explainability & Interpretability (2025-2026)](#explainability--interpretability-2025-2026)
    - [Reliability, Safety & Evaluation (2025-2026)](#reliability-safety--evaluation-2025-2026)
    - [RAG & Retrieval Methods (2024-2026)](#rag--retrieval-methods-2024-2026)
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

* [Python, ModernFinBERT](https://huggingface.co/tabularisai/ModernFinBERT) - Financial sentiment analysis model based on ModernBERT architecture (released July 2025), specialized for financial texts, earnings calls, and analyst reports. Improves accuracy by up to 48% over existing FinBERT models.

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

* [Kurdish Medical Corpus](https://aro.koyauniversity.org/index.php/aro/article/view/1088) - Specialized medical sentiment dataset for Kurdish text classification achieving 92% accuracy and 92% F1-score with multilingual BERT [[paper]](https://doi.org/10.14500/aro.11088) (Badawi, 2023).

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
    * [Grok 4](https://x.ai/grok) - Specialized in monitoring online sentiment and identifying emerging trends on social media with real-time X/Twitter integration. Excels at tracking public opinion, brand perception, and viral content (released July 2025)

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
    * [ModernFinBERT](https://huggingface.co/tabularisai/ModernFinBERT) - Latest financial sentiment model based on ModernBERT architecture, up to 48% accuracy improvement (July 2025)
    * [BloombergGPT](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) - 50B parameter LLM for financial NLP including sentiment analysis

* **Healthcare/Mental Health**
    * [MentalRoBERTa](https://huggingface.co/mental/mental-roberta-base) - RoBERTa-base model trained on mental health-related posts from Reddit, specialized for classifying depression, anxiety, PTSD, stress, suicidal ideation, and neutral content

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

## LLM Techniques for Sentiment Analysis

A comprehensive guide to applying Large Language Models to sentiment analysis using modern prompting, retrieval, and fine-tuning strategies.

### Prompt Engineering

Designing effective prompts is the fastest route to high-accuracy sentiment classification with LLMs—no retraining required.

#### Techniques

* **Zero-Shot Prompting** — Ask the model to classify sentiment directly with no examples. Surprisingly competitive on simple polarity tasks.
* **Few-Shot Prompting** — Prepend 3–8 labeled examples. GPT-4o with few-shot + CoT achieves **84.54% F1** (text classification) and **99% F1** (sentiment analysis) [[source]](https://aclanthology.org/2024.findings-naacl.246.pdf).
* **Chain-of-Thought (CoT)** — Instruct the model to reason step-by-step before producing a label. Boosts irony detection by up to **46%** on Gemini-1.5-flash [[paper]](https://arxiv.org/abs/2601.08302).
* **Multi-Chain CoT** — Aggregates multiple reasoning paths to resolve ambiguous sentiment cues [[paper]](https://www.mdpi.com/2076-3417/15/22/12225).
* **Domain Knowledge CoT (DK-CoT)** — Injects domain knowledge (e.g. financial terminology) into the reasoning chain before classification [[paper]](https://link.springer.com/article/10.1007/s10791-025-09573-7).
* **Self-Consistency** — Sample multiple completions and take the majority vote. Reduces variance caused by stochastic decoding.
* **Sentiment-Controlled Prompts** — Steer output emotion via prompt phrasing; few-shot with human-written examples is the most effective control strategy [[paper]](https://arxiv.org/abs/2602.06692).

#### Key Findings (2025-2026)

* GPT-4o **without** CoT outperforms all tested models on zero-shot financial sentiment (GPT-4o, GPT-4.1, o3-mini comparison) [[paper]](https://dl.acm.org/doi/10.1145/3768292.3770341).
* Negative prompts reduce factual accuracy and amplify bias; positive prompts increase verbosity [[paper]](https://arxiv.org/abs/2503.13510).
* Accuracy can fluctuate ±10% across identical runs — prompt stability matters as much as prompt design.

#### Tools & Guides

* [Prompt Engineering Guide — Sentiment Classification](https://www.promptingguide.ai/prompts/classification/sentiment)
* [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
* [Comprehensive Taxonomy of Prompt Engineering Techniques](https://link.springer.com/article/10.1007/s11704-025-50058-z)

---

### In-Context Learning & Few-Shot Methods

* **Zero-shot SLM Ensembles** — Combining multiple Small Language Models rivals proprietary LLMs at a fraction of the cost [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1566253525007389).
* **Multi-Agent LLMs** — Route different sentiment sub-tasks (coarse polarity, fine-grained emotion, irony) to specialist agents; demonstrated for social media in 2026 [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0342053).
* **LLM-Infused Multi-Module Transformer** — Injects LLM representations into a smaller model for few-shot emotion-aware sentiment [[paper]](https://www.sciencedirect.com/science/article/pii/S1566253525007407).

---

### Retrieval-Augmented Generation (RAG)

RAG grounds LLM sentiment predictions in external knowledge, reducing hallucinations and enabling domain-specific adaptation without retraining.

#### Architectures

* **Naive RAG** — Retrieve → Read → Generate. Baseline architecture; FAISS or Elasticsearch for retrieval.
* **Modular RAG** — Separate, swappable retrieval, reranking, and generation modules.
* **Self-RAG / Corrective RAG (CRAG)** — Model iteratively decides when to retrieve and critiques its own output before producing a label (2025).
* **Agentic RAG** — Embeds autonomous agents into the pipeline for planning, multi-hop retrieval, and tool use [[paper]](https://arxiv.org/abs/2501.09136).

#### Frameworks & Tools

* [LangChain](https://github.com/langchain-ai/langchain) — Modular LLM application framework; LCEL pipeline syntax makes sentiment pipelines composable. Introduced LangGraph for complex reasoning workflows (2025).
* [LlamaIndex](https://github.com/run-llama/llama_index) — Data framework for LLM apps; 300+ integrations, 35% retrieval accuracy boost in 2025. Best for document-heavy sentiment pipelines.
* [LangGraph](https://github.com/langchain-ai/langgraph) — Graph-based workflow orchestration for multi-step and agentic sentiment reasoning (2025).

#### Key Statistics (2025)

* 1,200+ RAG papers published on arXiv in 2024 alone vs. <100 in 2023.
* 63.6% of enterprise RAG deployments use GPT-based models.
* RAG evaluation survey: [[arXiv 2504.14891]](https://arxiv.org/abs/2504.14891).

---

### Parameter-Efficient Fine-Tuning (PEFT)

Fine-tune LLMs for sentiment without updating all parameters — dramatically reduces memory and compute.

#### Methods

* **LoRA (Low-Rank Adaptation)** — Freezes base weights, trains low-rank decomposition matrices. ~27–30 GB training memory.
    * LLaMA-3 + LoRA: **86.89% accuracy** on Financial PhraseBank [[paper]](https://link.springer.com/article/10.1007/s10586-025-05865-1).
    * [PEFT Library](https://github.com/huggingface/peft) (Hugging Face)

* **QLoRA (Quantized LoRA)** — Quantizes backbone to 4-bit, trains LoRA adapters. ~17–18 GB training memory. Enables 65B models on a single 48 GB GPU.
    * LLaMA-3 + QLoRA: **91.2% accuracy / 0.908 F1** on IMDB, **85.6% / 0.849 F1** on Twitter [[paper]](https://journals.adbascientific.com/csai/article/view/112).
    * QLoRA for Financial SA: up to **48% accuracy improvement** over baseline [[paper]](https://www.researchgate.net/publication/388119978_Sentiment_Analysis_with_LLMs_Evaluating_QLoRA_Fine-Tuning_Instruction_Strategies_and_Prompt_Sensitivity).
    * [QLoRA Repository](https://github.com/artidoro/qlora)

* **LoRAFusion** — Kernel-level QLoRA optimizations targeting 4-bit inference efficiency (EuroSys 2026) [[paper]](https://arxiv.org/html/2510.00206v1).

* **Multimodal LoRA** — Applies LoRA fine-tuning to vision-language LLMs (VLCLNet) for multimodal sentiment analysis [[paper]](https://dl.acm.org/doi/10.1145/3709147).

#### Tutorials

* [QLoRA Official Guide](https://arxiv.org/abs/2305.14314)
* [2025 Guide to Fine-Tuning: LoRA, QLoRA & Transfer Learning](https://medium.com/@dewasheesh.rana/the-ultimate-2025-guide-to-llm-slm-fine-tuning-sampling-lora-qlora-transfer-learning-5b04fc73ac87)
* [Keras: LoRA and QLoRA fine-tuning of Gemma](https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/)

---

### Instruction Tuning & Alignment

Aligning LLMs to produce correctly-formatted sentiment labels and reliable confidence scores.

#### Methods

* **Supervised Fine-Tuning (SFT)** — Trains on (instruction, sentiment-label) pairs to steer output format.
* **RLHF (Reinforcement Learning from Human Feedback)** — PPO-based training. SOTA on complex tasks; only 8% unsafe outputs under adversarial testing.
* **DPO (Direct Preference Optimization)** — Simpler, no reward model needed. Outperforms RLHF for sentiment-controlled generation. [[paper]](https://arxiv.org/abs/2305.18290)
* **RLAIF** — Replaces human annotators with an AI judge; scales sentiment preference data cheaply.

#### Key Survey

* [Comprehensive Survey of LLM Alignment: RLHF, RLAIF, PPO, DPO and More](https://arxiv.org/abs/2407.16216) (2024)

[Back to Top](#table-of-contents)

## LLM Evaluation & Benchmarks for Sentiment Analysis

### Benchmark Frameworks

* **SentiEval** — Proposed comprehensive LLM evaluation benchmark covering 13 SA task types on 26 datasets. Highlights gap between LLM and fine-tuned SLM on complex tasks. [[paper]](https://aclanthology.org/2024.findings-naacl.246/)
* **TruthfulQA** — Tests whether LLMs produce truthful answers; used to cross-reference hallucination rates in sentiment contexts.
* **HallucinationEval** — Dedicated benchmark for measuring LLM hallucination across NLP tasks including sentiment.
* **SemEval-2025 Task 10** — Multilingual characterization of subjectivity in news articles [[proceedings]](https://aclanthology.org/2025.semeval-1.331.pdf).
* **SemEval-2026 Task 3** — Dimensional Aspect-Based Sentiment Analysis on Customer Reviews (valence-arousal framework). Co-located with ACL 2026, San Diego. [[call for participation]](https://www.aclweb.org/portal/content/call-participation-semeval-2026-task-3-dimensional-aspect-based-sentiment-analysis-customer)

### Evaluation Metrics

| Metric | Use Case |
|--------|----------|
| Accuracy / F1 | Standard classification performance |
| Macro-F1 | Class-balanced evaluation (important for skewed SA datasets) |
| TARr@N / TARa@N | **Inference stability** — measures output variance across N identical runs |
| Confidence Calibration | Whether model confidence correlates with actual accuracy |
| ROUGE / BLEU | For rationale/explanation quality in generative SA |
| Perplexity | Language model fit on sentiment corpora |

### Model Performance Leaderboard (2025-2026)

| Model | Overall SA Accuracy | Notes |
|-------|--------------------|----|
| GPT-4o (few-shot + CoT) | ~99% F1 | Best on structured tasks [[NAACL 2024]](https://aclanthology.org/2024.findings-naacl.246.pdf) |
| Claude 3.7 | 79% | Best overall accuracy in 2025 benchmark |
| Claude 4.5 | 75% avg / 82% emotion detection | 2025 benchmark |
| GPT-4.1 | ~75–78% | Varies by domain |
| GPT-4o (zero-shot) | Best on financial SA | No CoT outperforms CoT variants [[paper]](https://dl.acm.org/doi/10.1145/3768292.3770341) |
| DeepSeek V3 | 70% | Competitive open-weight model |
| LLaMA-3 + QLoRA | 91.2% on IMDB / 85.6% Twitter | Fine-tuned, not zero-shot |

### Explainable Sentiment Analysis Dataset

* [Explainable Sentiment Analysis Dataset](https://ieee-dataport.org/documents/explainable-sentiment-analysis-dataset) — Released February 2025 on IEEE DataPort. Includes Amazon Reviews and IMDB, annotated with ground-truth sentiment labels, model predictions (GPT-4o, GPT-4o-mini, DeepSeek-R1), and fine-grained classifications for explainability evaluation.

[Back to Top](#table-of-contents)

## Explainable Sentiment Analysis

Understanding *why* a model produced a sentiment label — essential for production reliability, regulatory compliance, and debugging.

### Methods & Tools

#### Post-hoc Explanations

* **SHAP (SHapley Additive Explanations)**
    * Provides both **global** (feature importance across dataset) and **local** (single-prediction) explanations.
    * Applied layer-by-layer across LLM components (embedding → encoder → decoder → attention) for granular sentiment attribution.
    * [SHAP Library](https://github.com/slundberg/shap)
    * Recent benchmark: SHAP outperforms LIME on consistency and faithfulness [[paper]](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400304).

* **LIME (Local Interpretable Model-Agnostic Explanations)**
    * Perturbs input text and fits a local surrogate model to explain individual predictions.
    * Widely used to explain chatbot responses and customer sentiment decisions.
    * [LIME Library](https://github.com/marcotcr/lime)
    * Limitation: local explanations only; no global view.

* **ModernBERT-XAI** — Fine-tunes ModernBERT on IMDb and integrates SHAP + LIME for interpretable sentiment analysis. Released December 2025. [[paper]](https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2600795)

* **Attention Visualization**
    * Maps which tokens most influenced the sentiment decision.
    * Sentence-level attention visualization for LLMs: [[NAACL 2025 Demo]](https://aclanthology.org/2025.naacl-demo.27.pdf).

#### Causal and Counterfactual Methods

* **Counterfactual Testing** — Generate minimally-modified inputs that flip the sentiment label to identify causal features.
* **Causal Reasoning** — Grounding predictions in causal graphs reduces both bias and hallucination.

### Survey Papers

* [LLMs for Explainable AI: A Comprehensive Survey](https://arxiv.org/html/2504.00125v1) — April 2025. Covers how LLMs can themselves serve as explainers.
* [Integration of XAI Techniques with LLMs for Enhanced Interpretability for Sentiment Analysis](https://arxiv.org/abs/2503.11948) — March 2025.
* [SHAP and LIME: A Perspective on XAI Methods](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400304) — 2025.

### Practical Guides

* [SHAP + LIME for Real-Time Predictions in Production](https://www.javacodegeeks.com/2025/03/explainable-ai-in-production-shap-and-lime-for-real-time-predictions.html) — March 2025.
* [DataCamp: Explainable AI with LIME, SHAP, and InterpretML](https://www.datacamp.com/tutorial/explainable-ai-understanding-and-trusting-machine-learning-models)

[Back to Top](#table-of-contents)

## LLM Reliability & Safety in Sentiment Analysis

Critical considerations before deploying LLM-based sentiment classifiers in production.

### Hallucination

| Model | Hallucination Rate | Source |
|-------|--------------------|--------|
| GPT-4 | ~28.6% | medical systematic reviews |
| GPT-3.5 | ~39.6% | medical systematic reviews |
| Bard | ~91.4% | medical systematic reviews |
| Sentiment SA tasks | **Lower** | Pre-defined labels constrain generation |

**Mitigation Strategies:**
* **RAG** — Grounds predictions in retrieved evidence.
* **Multi-LLM Consensus** — Vote across 3+ models; agreement increases reliability.
* **Knowledge Graphs** — Inject structured facts at pretraining or inference time.
* **Self-Consistency Decoding** — Sample multiple completions, take majority.
* **Chain-of-Thought + Verification** — Have model verify its own reasoning step.

**Survey:** [Large Language Models Hallucination: A Comprehensive Survey](https://arxiv.org/html/2510.06265v2) (October 2025)

---

### Bias & Fairness

Five key bias-detection metrics applied to sentiment models:

1. **Counterfactual Testing** — Swap demographic attributes; check if sentiment label changes.
2. **Stereotype Detection** — Probe for systematically biased associations.
3. **Sentiment & Toxicity Analysis** — Measure polarity asymmetry across demographic groups.
4. **Acceptance/Rejection Rates** — Track differential response rates per group.
5. **Embedding-Based Metrics** — Measure cosine distance between group-specific embeddings.

**Survey:** [Bias in Large Language Models: Origin, Evaluation, and Mitigation](https://arxiv.org/html/2411.10915v1) (November 2024)
**Paper:** [Towards Trustworthy LLMs: Debiasing and Dehallucinating](https://link.springer.com/article/10.1007/s10462-024-10896-y) (2024)

---

### Uncertainty & Variability

* **Model Variability Problem (MVP)** — Identical prompts produce different sentiment labels across runs (up to ±10% accuracy).
* **Epistemic Uncertainty** — Model uncertainty due to lack of knowledge; mitigated by larger training sets or RAG.
* **Aleatoric Uncertainty** — Irreducible noise in ambiguous or contradictory sentiment texts.
* **Stability Metrics** — TARr@N and TARa@N measure inference stability across N runs [[paper]](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/full).

**Key Paper:** [Model Uncertainty and Variability in LLM-Based Sentiment Analysis: Challenges, Mitigation Strategies, and the Role of Explainability](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/full) — Frontiers in AI, 2025.

---

### Domain Instability

LLMs exhibit **12–18% higher accuracy degradation** on specialized domains (finance, healthcare, legal) vs. general text. Key causes:
* Technical jargon misinterpreted as neutral
* Sarcasm and irony patterns differ by domain
* Context-dependent sentiment cues absent from training data

**Mitigation:** Domain-specific fine-tuning (FinBERT, QF-LLM), knowledge-augmented prompting, domain-aware RAG.

**Paper:** [QF-LLM: Financial Sentiment Analysis with Quantized LLM](https://dl.acm.org/doi/10.1145/3764727.3764731) (2025)

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

### Prompt Engineering & LLM Methods (2025-2026)

* [Enhancing Sentiment Classification and Irony Detection through Advanced Prompt Engineering Techniques](https://arxiv.org/abs/2601.08302) — Evaluates few-shot, CoT, and self-consistency prompting; CoT boosts irony detection by **46%** on Gemini-1.5-flash. arXiv January 2026.

* [Evaluating Prompt Engineering Strategies for Sentiment Control in AI-Generated Texts](https://arxiv.org/abs/2602.06692) — Compares zero-shot, few-shot, and CoT for emotion steering; few-shot with human examples is most effective. arXiv February 2026.

* [Enhancing Granular Sentiment Classification with Chain-of-Thought Prompting in Large Language Models](https://arxiv.org/abs/2505.04135) — Focuses on fine-grained (multi-class) sentiment with CoT prompting. arXiv May 2025.

* [Prompt Sentiment: The Catalyst for LLM Change](https://arxiv.org/abs/2503.13510) — Shows prompt sentiment itself influences model accuracy: negative prompts reduce factual accuracy, positive prompts increase verbosity. arXiv March 2025.

* [Reasoning or Overthinking: Evaluating LLMs on Financial Sentiment Analysis](https://dl.acm.org/doi/10.1145/3768292.3770341) — GPT-4o, GPT-4.1, o3-mini comparison; GPT-4o without CoT achieves best performance. ACM AI in Finance 2025.

* [Leveraging LLM as News Sentiment Predictor: A Knowledge-Enhanced Strategy](https://link.springer.com/article/10.1007/s10791-025-09573-7) — Domain Knowledge CoT (DK-CoT) improves financial news sentiment prediction with GLM. Springer Discover Computing 2025.

* [Designing Multi-Agent LLMs for Fine-Grained User Sentiment Detection on Social Media](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0342053) — Routes sub-tasks to specialist agents; PLOS ONE February 2026.

* [Sentiment Analysis in the Era of Large Language Models: A Reality Check](https://aclanthology.org/2024.findings-naacl.246/) — Benchmarks LLMs across 26 datasets, 13 tasks; introduces SentiEval. NAACL Findings 2024.

* [Zero- and Few-Shot Prompting with LLMs: A Comparative Study with Fine-tuned Models for Bangla Sentiment Analysis](https://aclanthology.org/2024.lrec-main.1549/) — LREC-COLING 2024; compares Flan-T5, GPT-4, Bloomz.

* [Exploring Zero-Shot SLM Ensembles as an Alternative to LLMs for Sentiment Analysis](https://www.sciencedirect.com/science/article/abs/pii/S1566253525007389) — Small Language Model ensembles rival proprietary LLMs at lower cost. Information Fusion 2025.

* [LLM-Infused Multi-Module Transformer for Emotion-Aware Sentiment Analysis in Few-Shot Scenarios](https://www.sciencedirect.com/science/article/pii/S1566253525007407) — Injects LLM representations into a smaller model for efficient few-shot SA. Information Fusion 2025.

* [Multi-Chain of Thought Prompt Learning for Aspect-Based Sentiment Analysis](https://www.mdpi.com/2076-3417/15/22/12225) — Multi-path reasoning patterns for nuanced ABSA. Applied Sciences 2025.

* [QF-LLM: Financial Sentiment Analysis with Quantized LLM](https://dl.acm.org/doi/10.1145/3764727.3764731) — Quantization for cost-effective financial SA deployment. ACM AIDF 2025.

### Parameter-Efficient Fine-Tuning (2025-2026)

* [Parameter-Efficient Fine-Tuning of LLaMA Models for Financial Sentiment Classification](https://link.springer.com/article/10.1007/s10586-025-05865-1) — LLaMA-3 + LoRA achieves **86.89% accuracy** on Financial PhraseBank. Cluster Computing 2025.

* [Sentiment Analysis with LLMs: Evaluating QLoRA Fine-Tuning, Instruction Strategies, and Prompt Sensitivity](https://www.researchgate.net/publication/388119978_Sentiment_Analysis_with_LLMs_Evaluating_QLoRA_Fine-Tuning_Instruction_Strategies_and_Prompt_Sensitivity) — QLoRA delivers up to 48% accuracy improvement. 2025.

* [Benchmarking QLoRA-Fine-Tuned LLaMA and DeepSeek Models for Sentiment Analysis](https://journals.adbascientific.com/csai/article/view/112) — LLaMA-3 QLoRA: 91.2% IMDB, 85.6% Twitter F1. CSAI 2026.

* [Multimodal Large Language Model with LoRA Fine-Tuning for Multimodal Sentiment Analysis](https://dl.acm.org/doi/10.1145/3709147) — VLCLNet applies LoRA to vision-language LLMs for MABSA. ACM TIST 2025.

* [LoRAFusion: Efficient LoRA Fine-Tuning for LLMs](https://arxiv.org/html/2510.00206v1) — Kernel-level optimizations for 4-bit QLoRA (EuroSys 2026).

* [Analyzing LLaMA3 Performance on Classification Using LoRA and QLoRA Techniques](https://www.mdpi.com/2076-3417/15/6/3087) — Comprehensive LoRA vs QLoRA ablation study. MDPI Applied Sciences March 2025.

* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Original QLoRA paper enabling 65B parameter fine-tuning on a single 48 GB GPU.

### Explainability & Interpretability (2025-2026)

* [ModernBERT-XAI: Sentiment Analysis with Layer-Wise Learning and SHAP-LIME Interpretability](https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2600795) — Fine-tunes ModernBERT on IMDb with integrated SHAP + LIME. December 2025.

* [Integration of Explainable AI Techniques with LLMs for Enhanced Interpretability for Sentiment Analysis](https://arxiv.org/abs/2503.11948) — Applies SHAP layer-by-layer across LLM components. arXiv March 2025.

* [LLMs for Explainable AI: A Comprehensive Survey](https://arxiv.org/html/2504.00125v1) — LLMs as explainers; covers XAI methods across NLP tasks. arXiv April 2025.

* [A Perspective on Explainable AI Methods: SHAP and LIME](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400304) — Comparative evaluation of SHAP vs LIME faithfulness and consistency. Advanced Intelligent Systems 2025.

### Reliability, Safety & Evaluation (2025-2026)

* [Model Uncertainty and Variability in LLM-Based Sentiment Analysis: Challenges, Mitigation Strategies, and the Role of Explainability](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/full) — Introduces TARr@N stability metrics; Frontiers in AI 2025.

* [Large Language Models Hallucination: A Comprehensive Survey](https://arxiv.org/html/2510.06265v2) — GPT-4: 28.6%, GPT-3.5: 39.6% hallucination rates. arXiv October 2025.

* [Bias in Large Language Models: Origin, Evaluation, and Mitigation](https://arxiv.org/html/2411.10915v1) — Five bias-detection metrics applicable to sentiment models. arXiv November 2024.

* [Towards Trustworthy LLMs: Debiasing and Dehallucinating](https://link.springer.com/article/10.1007/s10462-024-10896-y) — Causal reasoning reduces both bias and hallucinations. AIR 2024.

* [Comparing LLMs and Human Annotators in Latent Content Analysis of Sentiment, Political Leaning, Emotional Intensity and Sarcasm](https://www.nature.com/articles/s41598-025-96508-3) — Multi-model comparison (GPT-3.5, GPT-4, GPT-4o, Llama-3.1, Mixtral). Scientific Reports 2025.

* [Evaluating LLMs for Sentiment Analysis on Vaccine Posts from Social Media](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526656/) — Healthcare/public health domain evaluation. JMIR Formative Research 2025.

### RAG & Retrieval Methods (2024-2026)

* [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) — Covers Self-RAG, Corrective RAG, and agent-driven retrieval pipelines for LLM applications. arXiv January 2025.

* [RAG Evaluation in the Era of LLMs: A Comprehensive Survey](https://arxiv.org/abs/2504.14891) — Reviews factual accuracy, safety, and computational efficiency metrics for RAG. arXiv April 2025.

* [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — The foundational RAG survey (2,000+ citations); covers naive, advanced, and modular RAG.

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
