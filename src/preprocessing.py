# src/preprocessing.py
import re
import numpy as np
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess a given text:
    - Lowercase
    - Remove non-alphanumeric characters
    - Tokenize
    - Remove stopwords
    """
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

