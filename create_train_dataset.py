import pandas as pd
import csv

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
    return df[df['RESPONSE_TEXT'].str.strip().str.len() > 0 & pd.notna(df['RESPONSE_TEXT'])]

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

def build_dictionary(data, course_to_label):
    """
    Returns a list of (label, review) tuples
    """
    label_review_pairs = []

    for course in course_to_label.keys():
        # Extract reviews for that course
        course_reviews = filter_course_rows(data, course)

        # Add each review to dictionary with correct label
        label = course_to_label[course]
        n = 0
        for review in course_reviews:
            cleaned_review = review.replace("\n", "")
            label_review_pairs.append((label, cleaned_review))

    return label_review_pairs

def create_tsv_file(data, output_filename):
    # Open the file in write mode and specify the delimiter as '\t' for TSV
    with open(output_filename, 'w', newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t')

        # Write the data from the list of tuples
        tsvwriter.writerows(data)

    print(f"TSV file '{output_filename}' has been created.")

def main():
    data = preprocess_text(load_data("course_evals.csv"))

    # 0 = easy, 1 = medium, 2 = hard; labels assigned based on poll data
    course_to_label = {"CS-106A" : 0, "CS-106B" : 1, "CS-107" : 2, "CS-109" : 2, "CS-103" : 1, "CS-161" : 2, "MATH-51" : 2}

    label_to_review = build_dictionary(data, course_to_label)

    create_tsv_file(label_to_review, "train_data.tsv")

if __name__ == "__main__":
    main()