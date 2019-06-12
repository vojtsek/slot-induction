from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
import sys

stemmer = EnglishStemmer()
lemmatizer = WordNetLemmatizer()
for line in sys.stdin:
    print(' '.join([stemmer.stem(tk) for tk in word_tokenize(line)]))
