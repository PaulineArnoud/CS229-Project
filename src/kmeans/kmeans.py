import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import gensim.downloader as api
from util import load_data, preprocess_text, remove_stopwords
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from itertools import combinations

# Function to load pre-trained word embeddings
def load_word_embeddings(embedding_type):
    word_vectors = api.load(embedding_type)
    return word_vectors

# Function to compute the sum of word embeddings for a list of texts
def compute_text_embedding(course_to_reviews, word_vectors):
    embeddings = {}
    for course, reviews in course_to_reviews.items():
        for review in reviews:
            words = review.split()

            embedding = np.zeros(300)  # Assuming 300-dimensional word embeddings
            for word in words:
                if word in word_vectors:
                    embedding += word_vectors[word]
            embeddings[course] = embedding
    return embeddings

# Function to create course embeddings
def create_course_embeddings(course_data, word_vectors, course_ids=None):
    if course_ids is None:
        course_ids = course_data['COURSE_UNIQUE_ID'].unique()
    
    course_reviews = course_data[course_data.apply(lambda row: any(course_id in row['COURSE_UNIQUE_ID'] for course_id in course_ids), axis=1)]
    
    # Clean course IDs to keep text between the first and last dash
    course_reviews['COURSE_UNIQUE_ID'] = course_reviews['COURSE_UNIQUE_ID'].str.split('-').str[1:-1].str.join('-')
    
    course_reviews = course_reviews.groupby('COURSE_UNIQUE_ID')['RESPONSE_TEXT'].apply(list).reset_index()    

    course_reviews['Embedding'] = compute_text_embedding(course_reviews['RESPONSE_TEXT'], word_vectors)
    return course_reviews

# Function to perform K-means clustering
def perform_clustering(course_embeddings_pca, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(np.vstack(course_embeddings_pca))
    return kmeans.labels_

# Function to visualize the clustering results (2D PCA space)
def visualize_clusters(course_embeddings_pca, cluster_labels, n_clusters, course_ids):    
    plt.figure(figsize=(8, 6))
    for cluster_label in range(n_clusters):
        cluster_data = course_embeddings_pca[cluster_labels == cluster_label]
        cluster_course_ids = np.array(course_ids)[cluster_labels == cluster_label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}', alpha=0.7)
        for i, course_id in enumerate(cluster_course_ids):
            plt.annotate(course_id, (cluster_data[i, 0], cluster_data[i, 1]))
    
    plt.title('Course Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

# Function to print course clusters
def print_course_clusters(course_reviews, cluster_labels, n_clusters):
    course_reviews['Cluster'] = cluster_labels
    for cluster_label in range(n_clusters):
        cluster_courses = course_reviews[course_reviews['Cluster'] == cluster_label]['COURSE_UNIQUE_ID'].tolist()
        print(f'Cluster {cluster_label}: {", ".join(cluster_courses)}')

def find_closest_courses(course_reviews, cluster_labels, n_clusters):
    course_reviews['Cluster'] = cluster_labels
    closest_courses = {}

    for cluster_label in range(n_clusters):
        cluster_courses = course_reviews[course_reviews['Cluster'] == cluster_label]
        cluster_ids = cluster_courses['COURSE_UNIQUE_ID'].tolist()
        
        # Compute pairwise cosine similarities between course embeddings
        similarities = {}
        for course1, course2 in combinations(cluster_ids, 2):
            emb1 = cluster_courses[cluster_courses['COURSE_UNIQUE_ID'] == course1]['Embedding'].values[0]
            emb2 = cluster_courses[cluster_courses['COURSE_UNIQUE_ID'] == course2]['Embedding'].values[0]
            similarity = 1 - cosine(emb1, emb2)
            similarities[(course1, course2)] = similarity
        
        # Find the three closest courses for each course
        closest_courses_in_cluster = {}
        for course in cluster_ids:
            closest_courses_for_course = sorted(cluster_ids, key=lambda x: similarities.get((course, x), -1), reverse=True)[:3]
            closest_courses_in_cluster[course] = closest_courses_for_course
        
        closest_courses.update(closest_courses_in_cluster)
    
    # Save the results to a text file
    with open('closest_courses.txt', 'w') as f:
        for course, closest in closest_courses.items():
            f.write(f'{course}: Closest Courses - {", ".join(closest)}\n')

def main(selected_course_ids=None):

    # Load course review data from a CSV file
    file_name = "data/course_evals.csv"
    course_data = load_data(file_name)

    # Preprocess the text data
    course_data = preprocess_text(course_data)

    # Load pre-trained word embeddings (Word2Vec)
    word_vectors = load_word_embeddings("word2vec-google-news-300")

    # Pass a list of selected course IDs to cluster, or None to cluster all course IDs
    # selected_course_ids = ["CS-106A-", "CS-107-", "CS-103-", "CS-161-"]
    # selected_course_ids = ["-CS-1", "-CS-2"]
    selected_course_ids = ["CS-106A-", "CS-106B-", "CS-107-", "CS-109-", "CS-103-", "CS-161-", "MATH-51-", "CS-221-", "CS-229-", "CS-231", "CS-224", "CS-131", "CS-124", "CS-140-","CS-142-", "CS-238-", "CS-148-", "MATH-20-", "MATH-21-", "MATH-19-", "PHYSICS-41-", "PHYSICS-43-", "PHYSICS-21-", "PHYSICS-23-"]


    # Create course embeddings for selected course IDs or all course IDs
    course_reviews = create_course_embeddings(course_data, word_vectors, selected_course_ids)

    # Extract course IDs
    course_ids = course_reviews['COURSE_UNIQUE_ID'].unique()

    # Apply PCA for dimensionality reduction
    n_components = 2  # You can adjust the number of components as needed
    pca = PCA(n_components=n_components)
    course_embeddings_pca = pca.fit_transform(np.vstack(course_reviews['Embedding']))

    # Perform K-means clustering
    n_clusters = 3  # Number of clusters (easy, medium, hard)
    cluster_labels = perform_clustering(course_embeddings_pca, n_clusters)

    # Visualize the clustering results with course IDs
    visualize_clusters(course_embeddings_pca, cluster_labels, n_clusters, course_ids)

    # Print the course clusters
    print_course_clusters(course_reviews, cluster_labels, n_clusters)

    # Find the three closest courses for each course in the same cluster
    find_closest_courses(course_reviews, cluster_labels, n_clusters)

if __name__ == "__main__":
    main()
