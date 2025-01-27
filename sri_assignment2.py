# -*- coding: utf-8 -*-
"""Sri_Assign2.ipynb

# Basics of Text Pre-processing using NLTK.

Installing neccessary libraries and packages, and Importing them to use in the notebook,
"""

pip install nltk==3.5

pip install numpy matplotlib

from nltk.tokenize import sent_tokenize, word_tokenize

import nltk

nltk.download('punkt')

"""## Tokenizing
The first step in converting the unstructered data into structured data, We split the the text into either tokens of sentence or word.

"""

from nltk.tokenize import sent_tokenize, word_tokenize

Myself_string = """ Hi Everyone this is Sri Nikhitha Bokka.
... You all can call me Sri or Nikki.
... Currently I am working in software industry
... and doing masters in AI."""

#Tokenizing by Sentence
sent_tokenize(Myself_string)

#Tokenizing by Word
word_tokenize(Myself_string)

"""## Filtering Stop Words

We don't want 'in', 'is', ''s' to be a token, hence we will have Stop Words the words to ignore, filter them out of the text.
"""

nltk.download("stopwords")

from nltk.corpus import stopwords

words = word_tokenize(Myself_string)

words

"""Here we are using the same set of stop words that are standardily defined in the StopWords library and filtering them from the output of Tokeninzing by Words."""

stop_words = set(stopwords.words("english"))

#List of default stop words defined in NLTK
print(stop_words)

filtered_list = []

for word in words:
    if word.casefold() not in stop_words:
         filtered_list.append(word)

filtered_list = [
...     word for word in words if word.casefold() not in stop_words
... ]

filtered_list

"""## Stemming
Task where we reduce the words in the data to their roots. Following the tutorial, we are using Porter Stemmer from NLTK, A word stemmer based on Porter stemming Algorithm.
"""

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words

stemmed_words = [stemmer.stem(word) for word in words]

stemmed_words

"""'everyon',
 'thi', 'softwar',
 'industri',  These are results of stemming going wrong, There can false negatives and false positives.

## Tagging Parts of Speech
POS tagging is the part of preprocessing that describes for itself, We label the words in the text according to their part of speech.
"""

nltk.download('averaged_perceptron_tagger')

nltk.download('tagsets')

"""below comman shows the full form and meanings of Parts of speech"""

nltk.help.upenn_tagset()

nltk.pos_tag(words)

"""What if the sentence have gibberish words ?"""

jabberwocky_excerpt = """
... 'Twas brillig, and the slithy toves did gyre and gimble in the wabe:
... all mimsy were the borogoves, and the mome raths outgrabe."""

words_in_excerpt = word_tokenize(jabberwocky_excerpt)

nltk.pos_tag(words_in_excerpt)

"""It is interesting how NLTK has categorized few unknown gibberish words into POS.

## Lemmatization

Same as Stemming, Reducing words to their core meaning.
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer_sri = WordNetLemmatizer()

"""Let's test and see how lemmatizing words for few words,"""

lemmatizer_sri.lemmatize("running")

lemmatizer_sri.lemmatize("scarves")

""" The difference between lemmatizing and stemming is visible as lemmatizing returns whole words avoiding false negatives or false positives, Now lets do lemmatizing using POS taggings(parts of speech)"""

from nltk import pos_tag

tags = pos_tag(words)

"""Functuion to map pos tag to wordnet pos tag"""

def get_wordnet_pos(tag):
    if tag.startswith('Verb'):
        return wordnet.VERB
    elif tag.startswith('Noun'):
        return wordnet.NOUN
    elif tag.startswith('Adjective'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN

"""Lemmatizing words using POS tag"""

from nltk.corpus import wordnet

lemmatized_pos_string = [
    lemmatizer_Sri.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tags
]

print("Lemmatized sentence:", " ".join(lemmatized_pos_string))

lemmatized_with_pos = [(word, tag, get_wordnet_pos(tag)) for word, tag in tags]

lemmatized_string_with_pos = [
    (lemmatizer_Sri.lemmatize(word, pos=pos), tag, pos) for word, tag, pos in lemmatized_with_pos
]

for word, tag, pos in lemmatized_string_with_pos:
    print(f"Word: {word}, POS Tag: {tag}, WordNet POS: {pos}")

"""## Chunking

Chunking group words using POS tags and apply chunk tags to those groups.
"""

from nltk.tokenize import word_tokenize

String_quote = "It's a dangerous business, Frodo, going out your door."

words_in_String_quote = word_tokenize(String_quote)

words_in_String_quote

nltk.download("averaged_perceptron_tagger")

String_pos_tags = nltk.pos_tag(words)

String_pos_tags

"""Chunk grammer expression"""

grammar_expression = "NP: {<DT>?<JJ>*<NN>}"

chunk_parser = nltk.RegexpParser(grammar_expression)

tree = chunk_parser.parse(String_pos_tags)

tree.draw()

#tree.draw() doesn't work in google colab
print(tree.pretty_print())

"""## Chinking

Chinking is used together with chunking, but while chunking is used to include a pattern, chinking is used to exclude a pattern.
"""

String_pos_tags

grammar_expression = """
... Chunk: {<.*>+}
...        }<JJ>{"""

chunk_parser = nltk.RegexpParser(grammar_expression)

tree = chunk_parser.parse(String_pos_tags)

print(tree.pretty_print())

"""##Named Entity Recognition (NER)

Named entity recognition let's you to find named entities in your texts and determine what kind of named entity they are like Date, Person, Time etc.
"""

nltk.download("maxent_ne_chunker")

nltk.download("words")

tree = nltk.ne_chunk(String_pos_tags)

print(tree.pretty_print())

tree = nltk.ne_chunk(String_pos_tags, binary=True)

print(tree.pretty_print())

"""Function to extract named entities in our string

"""

def extract_ne(quote):
...     words = word_tokenize(quote, language=language)
...     tags = nltk.pos_tag(words)
...     tree = nltk.ne_chunk(tags, binary=True)
...     return set(
...         " ".join(i[0] for i in t)
...         for t in tree
...         if hasattr(t, "label") and t.label() == "NE"
...     )

#Named entities in our string:
extract_ne(Myself_string)

def extract_ne(quote, language='English'):
...     words = word_tokenize(quote, language=language)
...     tags = nltk.pos_tag(words)
...     tree = nltk.ne_chunk(tags, binary=True)
...     return set(
...         " ".join(i[0] for i in t)
...         for t in tree
...         if hasattr(t, "label") and t.label() == "NE"
...     )

extract_ne(Myself_string)

import nltk

nltk.download('punkt')

def extract_ne(quote, language='english'):
...     words = word_tokenize(quote, language=language)
...     tags = nltk.pos_tag(words)
...     tree = nltk.ne_chunk(tags, binary=True)
...     return set(
...         " ".join(i[0] for i in t)
...         for t in tree
...         if hasattr(t, "label") and t.label() == "NE"
...     )

extract_ne(Myself_string)

"""# Corpora Analysis

In this section, we will be using few corpus from NLTK.
"""

#Downloading all the text data in book of NLTK
nltk.download("book")

from nltk.book import *

"""## Concordance

This can give you a peek into how a word is being used at the sentence level and what words are used with it.
"""

text1.concordance("man")

text4.concordance("woman")

#Just curious
text1.concordance("Nikki")
text2.concordance("Nikki")
text3.concordance("Nikki")
text4.concordance("Nikki")
text5.concordance("Nikki")
text6.concordance("Nikki")
text7.concordance("Nikki")
text8.concordance("Nikki")
text9.concordance("Nikki")

"""## Dispersion Plot
A dispersion plot is used to see how much a particular word appears and where it appears in the text, Here I would like see the plots of all the texts in the book collection of NLTK
"""

text1.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text2.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text3.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text4.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text5.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text6.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text7.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text8.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )
text9.dispersion_plot(
...     ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
... )

"""## Frequency Distribution

Checking which words are most used in the text using FreqDist subclass of collections.Counter.
"""

from nltk import FreqDist

frequency_distribution = FreqDist(text8)

print(frequency_distribution)

frequency_distribution.most_common(15)

meaningful_words = [
...     word for word in text8 if word.casefold() not in stop_words
... ]

frequency_distribution = FreqDist(meaningful_words)

frequency_distribution.most_common(15)

frequency_distribution.plot(15, cumulative=True)

"""## Collocations

Most common sequence of words in the text data.
"""

text2.collocations()

"""It would be interesting to see lemmatized colloactions in the text2 from book collection"""

lemmatized_words = [lemmatizer.lemmatize(word) for word in text2]

new_text = nltk.Text(lemmatized_words)

new_text.collocations()

"""# Conclusion

The text pre-processing using NLTK involves the steps involved in the NLP pipeline. By following the tutorial and reading the textbook material, I could map the the steps in the tutorial to the steps involved in the pipeline. As per my understanding, Filtering stop words, lemmatizing, and tokenizing are all part of lexical analysis, which divides text into tokens. Lemmatizing, stemming, and filtering stop words are all part of morphological analysis. Identifying parts of speech, chunking, and chinking are all part of syntactic analysis. Recognizing named entities is a component of semantic analysis. A dispersion plot is created as part of discourse integration. Pragmatic analysis uses collocations and frequency distribution.

The way we discover the NLP pipeline by several steps in the tutorial has increased my understanding of the pipeline. Using NLTK for the text preprocessing has been rewarding and I believe it's great for research, educational purposes, and prototyping smaller NLP models.

"""