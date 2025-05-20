
Emotion Classification in Tweets Using Support Vector Machines

Description:
This project builds a machine learning model to classify emotions expressed in English-language tweets. The model uses Support Vector Machines (SVM) to identify six basic emotions: anger, fear, joy, love, sadness, and surprise.

Dataset:
The dataset contains English tweets labeled with one of six emotions. Each row represents a tweet and its corresponding emotion label.

Project Steps:

1. Data Preprocessing:

   * Removed noise such as URLs, mentions, and special characters
   * Converted text to lowercase
   * Tokenized and vectorized text using TF-IDF with a maximum of 10,000 features

2. Model Building:

   * Trained a Support Vector Machine classifier using LinearSVC
   * Also attempted SVC with RBF kernel for comparison (optional and computationally expensive)

3. Evaluation:

   * Evaluated using accuracy and classification report
   * LinearSVC achieved approximately 89% accuracy
   * Model performed strongly on emotions like joy, sadness, and anger

Dependencies:

* Python 3.x
* pandas
* scikit-learn

Instructions to Run:

1. Install the required Python packages:
   pip install pandas scikit-learn

2. Place the dataset file emotions.csv in the same directory as the script.

3. Run the Python script to train the model and view the results.

Conclusion:
The LinearSVC model trained on TF-IDF features is effective for classifying emotions in tweets. It offers high accuracy and fast training time. This approach can be useful for real-world applications such as sentiment analysis, customer feedback analysis, and mental health monitoring.


