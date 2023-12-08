import pandas as pd
import csv
import random
import pickle

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
    train_pairs = []
    eval_pairs = []
    test_pairs = []

    for course in course_to_label.keys():
        # Extract reviews for that course
        course_reviews = filter_course_rows(data, course)

        # Add each review to dictionary with correct label
        label = course_to_label[course]
        n = 0
        for review in course_reviews:
            cleaned_review = review.replace("\n", "")

            n = random.random()
            if n < 0.7:
                train_pairs.append((label, cleaned_review))
            elif 0.7 <= n <= 0.9:
                eval_pairs.append((label, cleaned_review))
            else:
                test_pairs.append((label, cleaned_review))

    return train_pairs, eval_pairs, test_pairs

def create_tsv_file(data, output_filename):
    # Open the file in write mode and specify the delimiter as '\t' for TSV
    with open(output_filename, 'w', newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t')

        # Write the data from the list of tuples
        tsvwriter.writerows(data)

    print(f"TSV file '{output_filename}' has been created.")

def main():
    # preprocess data 
    data = preprocess_text(load_data("../data/course_evals.csv"))

    # for selected courses, create dict from each review to corresponding course id and save as pickle
    selected_course_ids = ["CS-106A", "CS-106B", "CS-107", "CS-109", "CS-103", "CS-161", "MATH-51"]
    review_to_course_id = {}
    for course_id in selected_course_ids:
        filtered_data = data[data['COURSE_UNIQUE_ID'].str.contains(course_id)]
        reviews = filtered_data['RESPONSE_TEXT'].tolist()
        for review in reviews:
            review_to_course_id[review] = course_id
    
    # 0 = easy, 1 = medium, 2 = hard; labels assigned based on poll data
    course_to_label = {"CS-106A" : 0, "CS-106B" : 1, "CS-107" : 2, "CS-109" : 2, "CS-103" : 1, "CS-161" : 2, "MATH-51" : 2}

    # get (label, review) pairs for training, eval, and test sets
    train_pairs, eval_pairs, test_pairs = build_dictionary(data, course_to_label)

    # save (label, review) pairs as tsv files
    create_tsv_file(train_pairs, "../data/train_data.tsv")
    create_tsv_file(eval_pairs, "../data/eval_data.tsv")
    create_tsv_file(test_pairs, "../data/test_data.tsv")

    # create dict from course id to indices of reviews in test dataset
    test_df = pd.read_csv('../data/test_data.tsv', delimiter='\t')
    course_to_review_indices_test = {course_id: [] for course_id in selected_course_ids}
    for index, row in test_df.iterrows():
        message = row[1]
        if message in review_to_course_id and review_to_course_id[message] in selected_course_ids:
            course_id = review_to_course_id[message]
            course_to_review_indices_test[course_id].append(index)

    with open('../data/course_to_review_indices_test.pkl', 'wb') as file:
        pickle.dump(course_to_review_indices_test, file)

    


if __name__ == "__main__":
    main()