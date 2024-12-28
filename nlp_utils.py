import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
nltk.download('punkt')   # This model helps split text into sentences or words based on language rules (like punctuation and spacing).
nltk.download('stopwords')  # Removing stopwords from a text to reduce noise in data.
nltk.download('wordnet')  # Used for lemmatization (converting words to their base or dictionary form) and semantic analysis.

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Function to preprocess storylines
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation, numbers, and special characters using regex
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    #  Lemmatization is the process of reducing a word to its base or dictionary form (i.e., its lemma). For example:
    # "running" -> "run"
    # "better" -> "good"

    # Join the tokens back into a single string
    return ' '.join(tokens)