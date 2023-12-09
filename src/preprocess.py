import pandas as pd
from nltk.corpus import stopwords, words as nltk_words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
from wordcloud import STOPWORDS as wordcloud_stopwords
import string

# Function to load data from a CSV file
def load_data(file_name):
    """
    Read the CSV file into a Pandas DataFrame

    Args:
        file_name (str): Name of the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data.
    """
    return pd.read_csv(file_name, encoding='latin1')

# Function to preprocess text data
def preprocess_text(df):
    """
    Filter out rows with no text in the "RESPONSE_TEXT" column or NaN values (i.e., no course review).

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with filtered data.
    """
    # Filter out rows with no text in the "RESPONSE_TEXT" column or NaN values (i.e., no course review). 
    filtered_rows = df[df['RESPONSE_TEXT'].str.strip().str.len() > 0 & pd.notna(df['RESPONSE_TEXT'])]
    return filtered_rows

# Function to filter rows for a specific course
def filter_course_rows(dataframe, course_name):
    """
    Filter rows in the DataFrame for a specific course.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.
        course_name (str): Name of the course to filter.

    Returns:
        pandas.DataFrame: Filtered DataFrame for the specified course.
    """
    course_rows = dataframe[dataframe['COURSE_UNIQUE_ID'].str.contains(course_name)]
    reviews_list = course_rows["RESPONSE_TEXT"].tolist()
    return reviews_list

# Function to remove stopwords and punctuation from a list of words
def remove_stopwords(word_list):
    # Load NLTK stopwords, English words, and punctuation
    stop_words = set(stopwords.words('english'))
    stop_words.update(nltk_words.words())
    stop_words.update(ENGLISH_STOP_WORDS)
    stop_words.update(gensim_stopwords)
    stop_words.update(wordcloud_stopwords)
    stop_words.update(string.punctuation)

    # Add a custom list to the stopwords
    custom_words = ["class", "course", "cs", "professor"]
    stop_words.update(custom_words)

    # Remove stopwords and punctuation
    filtered_words = [word for word in word_list if word.lower() not in stop_words]

    return filtered_words

def build_dicts(course_data, selected_course_ids):
    messages = []
    labels = []
    courses = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
            messages.append(message)
            labels.append(int(label))  # Convert label to integer

    dataset = Dataset.from_dict({"label": labels, "text": messages})
    return dataset

def main():
    # Load course review data from a CSV file
    file_name = "course_evals.csv"
    course_data = load_data(file_name)

    # Preprocess the text data
    course_data = preprocess_text(course_data)
    selected_course_ids = ["CS-106A-", "CS-106B-", "CS-107-", "CS-109-", "CS-103-", "CS-161-", "MATH-51-", "CS-221-", "CS-229-", "CS-231", "CS-224", "CS-131", "CS-124", "CS-140-","CS-142-", "CS-238-", "CS-148-", "MATH-20-", "MATH-21-", "MATH-19-", "PHYSICS-41-", "PHYSICS-43-", "PHYSICS-21-", "PHYSICS-23-"]
    course_data = build_dicts(course_data, selected_course_ids)
