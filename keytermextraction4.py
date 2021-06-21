from lxml import etree
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_normalized_tokens(text, stopword_list, punctuation_list, lemmatizer):
    text = text.lower()
    tokens = word_tokenize(text)
    new_tokens = []
    for token in tokens:
        token = lemmatizer.lemmatize(token)  
        if token not in stopword_list and token not in punctuation_list:
            pos = nltk.pos_tag([token])[0][1]
            if pos == "NN":
                new_tokens.append(token) 
    return new_tokens

def build_frequency_tuple(text, stopword_list, punctuation_list, lemmatizer):
    tokens = get_normalized_tokens(text, stopword_list, punctuation_list, lemmatizer)
    frequency_dict = {}
    for token in tokens:
        if token in frequency_dict:
            frequency_dict[token] += 1
        else:
            frequency_dict[token] = 1
    frequency_tuple = sorted(frequency_dict.items(), key=lambda x: (x[1], x[0]))
    frequency_tuple.reverse()
    return frequency_tuple

def analize_frequency(corpus):
    for news in corpus:
        title = news[0].text
        text = news[1].text
        frequency_tuple= build_frequency_tuple(text, stopword_list, punctuation_list, lemmatizer)
        print_frequency(title, frequency_tuple)
        print()

def print_frequency(title, frequency_tuple):
    print(title + ":")
    five_list = []
    for tuple in frequency_tuple[:5]:
        word, frequency = tuple
        five_list.append(word)
    print(" ".join(five_list))

def print_important(title, wordlist):
    print(title + ":")
    five_list = []
    for tuple in wordlist[:5]:
        word, tfidf = tuple
        five_list.append(word)
    print(" ".join(five_list))

def build_normalize_text(text, stopword_list, punctuation_list, lemmatizer):
    tokens = get_normalized_tokens(text, stopword_list, punctuation_list, lemmatizer)
    text = " ".join(tokens)
    return text

def build_important_tuple(titles, docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names()
    for i, title in enumerate(titles):
        row = tfidf_matrix.getrow(i)
        m = row.tocoo()
        rows = m.row
        cols = m.col
        data = m.data
        wordlist = []
        for i, col in enumerate(cols):
            tuple = terms[col], data[i]
            wordlist.append(tuple)
        wordlist.sort(key=lambda x: (x[1], x[0]), reverse=True)
        print_important(title, wordlist)
        print()

lemmatizer = WordNetLemmatizer()
stopword_list = stopwords.words('english')
punctuation_list = list(string.punctuation)
parser = etree.XMLParser(recover=True)
tree = etree.parse('news.xml', parser)
data = tree.getroot()
corpus = data[0]
titles = []
docs = []
for news in corpus:
    title = news[0].text
    text = news[1].text
    text = build_normalize_text(text, stopword_list, punctuation_list, lemmatizer)
    titles.append(title)
    docs.append(text)

build_important_tuple(titles, docs)
