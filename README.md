# CS229-Project

## Project Motivation and Overview

A common pain point among university students is course selection, a decision influenced by a myriad of factors including syllabi, professor reputation, and subjective peer reviews. A course recommended by a student as ”a walk in the park” may end up consuming another student’s entire quarter. The aim of this project is to provide a data-driven approach for accurately assessing course difficulty, thus aiding students in making informed decisions for a successful educational journey. By offering a clear, objective metric for course difficulty, we intend to streamline the decision-making process and improve the academic experience.

The input data is student course reviews from Stanford's course exploration tool Carta. We then apply various ML techniques, including Naive Bayes, BERT, K-means, and sentiment analysis, to output a predicted difficulty rating (easy, medium, hard) for each course.

## Data

Our dataset is sourced from the Carta platform managed by Stanford’s Pathways Lab, which includes course review data for all Stanford courses. We focused on the Winter 2023 course review data. Each dataset row represents a student’s
course review, containing CLASS ID, STUDENT ID, and responses from Stanford’s official course evaluations (see 'course_evals.csv'). In addition to course reviews, we got labels for the difficulty of courses by gathering data on how students perceived the difficulty of specific courses. We distributed a Google Forms survey widely to Stanford students, ensuring representation across academic years and majors

After obtaining our difficulty labels, we organized the data into a CSV file, with each row representing a review for one of 7 selected courses. Each review was assigned the difficulty label attributed to its corresponding course. This
yielded a total of 1,655 reviews, with each review serving as a unique sample within our dataset. We divided them using the standard ratio of 80% for the training set (see 'train_data.csv'), 20% the testing set (see 'test_data.csv'). 

## Methods

### Classification
We classified the difficulty of individual course reviews in addition to entire courses. While the main objective of our project is to determine the difficulty of a course, we decided to classify the difficulty of individual course reviews because our dataset has significantly more data points for reviews (n = 1655) than courses (n = 7) and thus produces more robust results for evaluating the models. For every classification model, we first classified reviews and then classified the difficulty of each course by retrieving the most frequent predicted difficulty among reviews associated with a given course.

### Naive Bayes 
We selected the Naive Bayes Classifier as our baseline due to its proven effectiveness in text classification, simplicity, and our familiarity with it from coursework. All code for its implementation is in 'src/naivebayes'. 

### BERT  
To address the weakness of our Naive Bayes model in capturing contextual dependencies between words, we turned to Bidirectional Encoder Representations from Transformers (BERT). We fine-tuned a pre-trained BERT model
called ”DistilBERT” on our course review dataset, adapting it to classify course reviews into easy, medium, or hard categories based on their content All code for its implementation is in 'src/bert'. 

### K-Means
Our third approach attempts to address the limitation of both the Naive Bayes and DistilBERT models, which is that individual reviews’ contents may not necessarily align with the overall course’s label. Although k-means clustering in itself doesn’t address the first limitation of the naive assumption, we paired it with vector embeddings (as outlined in the ”Dataset & Features” section) to be able to group semantically similar reviews together. We employed k-means clustering to group courses with comparable review content into distinct clusters. Each cluster represented a set of courses with akin attributes, providing an alternative perspective on course difficulty based on content similarity. All code for its implementation is in 'src/kmeans'. 

### Sentiment
Our final approach takes a deeper probe at the  limitation of the Naive Bayes model, that individual reviews’ contents may not necessarily align with the overall course’s label. First, we use a popular LLM fine-tuned on sentiment analysis to generate positive or negative sentiment scores from 0 to 1 (0 for weaker emotions, 1 for strong emotions) for each review text. We then average those scores to come up with an aggregate sentiment score for each course.
We then do two things with these scores. First, we run kmeans clustering using these sentiment scores instead of word2vec embeddings as in our second approach. Second, we attempt to translate those sentiment scores into difficulty
labels. All code for its implementation is in 'src/sentiment'
