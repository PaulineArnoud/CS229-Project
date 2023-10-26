import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords, words as nltk_words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
from wordcloud import STOPWORDS as wordcloud_stopwords
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to download NLTK resources
def download_nltk_resources():
    """
    Download NLTK resources (stopwords and words data) if not already downloaded.
    """
    nltk.download('stopwords')
    nltk.download('words')  # Download the words in the English dictionary

# Function to preprocess text data
def preprocess_text(df):
    """
    Filter out rows with no text in the "RESPONSE_TEXT" column or NaN values (i.e., no course review).

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with filtered data.
    """
    return df[df['RESPONSE_TEXT'].str.strip().str.len() > 0 & pd.notna(df['RESPONSE_TEXT'])]

# Function to extract common words for a specific course
def extract_common_words(dataframe, course_name, custom_stopwords, color_palette):
    """
    Extract the most common words for a specific course and return them.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.
        course_name (str): Name of the course.
        custom_stopwords (list): List of custom stopwords to exclude.
        color_palette (seaborn.Palette): Seaborn color palette for plotting.

    Returns:
        list: List of the most common words for the course.
    """
    course_df = filter_course_rows(dataframe, course_name)
    word_frequency_dict = build_word_dictionary(course_df, custom_stopwords)
    most_common_words = word_frequency_dict.most_common(10)
    return most_common_words

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
    return dataframe[dataframe['COURSE_UNIQUE_ID'].str.contains(course_name)]

# Function to build a word dictionary
def build_word_dictionary(dataframe, custom_stopwords):
    """
    Build a dictionary of words from the "RESPONSE_TEXT" column of a DataFrame.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.
        custom_stopwords (list): List of custom stopwords to exclude from the dictionary.

    Returns:
        collections.Counter: Dictionary mapping words to their frequencies.
    """

    # Initialize a word frequency counter
    word_counts = Counter()

    # Define words in the english dictionary
    english_words = set(nltk_words.words())

    # Define stop words from different sources
    stop_words = set(stopwords.words('english'))
    stop_words.update(ENGLISH_STOP_WORDS)
    stop_words.update(gensim_stopwords)
    stop_words.update(wordcloud_stopwords)

    # Merge the provided list of custom stop words with the combined stop words list, if provided
    if custom_stopwords is not None:
        stop_words.update(custom_stopwords)

    # Iterate through the "RESPONSE_TEXT" column, tokenize, and count words
    for response in dataframe['RESPONSE_TEXT'].dropna():
        words = re.findall(r'\w+', response.lower())
        words = [word for word in words if word not in stop_words and word in english_words]
        word_counts.update(words)

    return word_counts

# Function to plot common words for each course
def plot_common_words_per_course(course_list, common_words_dict, color_palette):
    """
    Plot the most common words for each course using a specified color palette.

    Args:
        course_list (list): List of course names.
        common_words_dict (dict): Dictionary containing common words for each course.
        color_palette (seaborn.Palette): Seaborn color palette for plotting.
    """

    plots_per_row = 5
    num_coursees = len(course_list)
    num_rows = -(-num_coursees // plots_per_row)

    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, num_rows * 4))
    plt.subplots_adjust(hspace=0.5)

    for i, course_name in enumerate(course_list, start=1):
        most_common_words = common_words_dict[course_name]

        if most_common_words:
            words, counts = zip(*most_common_words)
            row, col = divmod(i - 1, plots_per_row)
            color = color_palette[i % len(color_palette)]
            axs[row, col].bar(words, counts, color=color)
            axs[row, col].set_ylabel("Frequency")
            axs[row, col].set_title(f"Top Words in {course_name}")
            axs[row, col].tick_params(axis='x', rotation=45, labelsize=8)
        else:
            # Handle the case when there are no most_common_words
            row, col = divmod(i - 1, plots_per_row)
            axs[row, col].text(0.5, 0.5, f"{course_name} not offered Win 21-22", ha='center', va='center', fontsize=10, color='gray')
            axs[row, col].axis('off')

    for i in range(len(course_list), num_rows * plots_per_row):
        row, col = divmod(i, plots_per_row)
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.savefig('wordcloud_plots.png')

def main():
    # Configuration
    file_name = "course_evals.csv"
    course_list = ["CS-106A", "CS-106B", "CS-107", "CS-109", "CS-103", "CS-161", "CS-110", "MATH-51", "MATH-19", "MATH-20", "MATH-21",
                  "PHYSICS-41", "PHYSICS-43", "PHYSICS-21", "PHYSICS-23", "ECON-1", "ECON-50", "ENGR-40M", "CHEM-31B"]
    custom_stopwords = ["class", "course", "cs", "course", "professor", "physics", "econ"]
    color_palette = sns.color_palette("dark")

    # Load data
    df = load_data(file_name)

    # Download NLTK resources
    download_nltk_resources()

    # Preprocess data
    reviews_with_text_df = preprocess_text(df)

    # Extract common words for each course
    common_words_dict = {}
    for course_name in course_list:
        most_common_words = extract_common_words(reviews_with_text_df, course_name, custom_stopwords, color_palette)
        common_words_dict[course_name] = most_common_words

    # Plot common words for each course
    plot_common_words_per_course(course_list, common_words_dict, color_palette)

if __name__ == "__main__":
    main()