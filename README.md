# Toxic Comment Analysis
DS 4021: Machine Learning II Final Project

Dongju Han, Kayla Kim, Mason Nicoletti

## Introduction:

Social media harm is a pervasive issue, driven by the rapid spread of toxic comments, hate speech, and polarized discourse across online platforms. As part of Data Project 3, the team collected and analyzed large-scale Bluesky Firehose data, accessible in the project repository under the skyblue directory: [Skyblue Directory](https://github.com/djhan0330/ds3022-data-project-3/tree/main/skyblue)


Through this exploratory analysis, the team identified patterns and linguistic features indicative of harmful online content. To study these patterns rigorously, the project applies a range of machine-learning methods for sentiment and toxicity classification, including Support Vector Machines (SVM), logistic regression, ensemble methods such as Random Forest, and neural networks. By comparing the performance of these models, the analysis aims to provide deeper insight into how toxic speech manifests within social media data and which approaches are most effective at detecting harmful content at scale.

## Software and Platform Requirements

To run the notebooks and scripts in this project, you will need:
* pandas>=2.0
* numpy>=1.26
* scikit-learn>=1.4
* scipy>=1.10
* matplotlib>=3.8
* seaborn>=0.13
* wordcloud>=1.9
* tqdm>=4.66
* torch (Only for Neural Network)

A full dependency list is provided in requirements.txt

## Repository Structure 

Below is an outline of the folders and files in this repository to help users quickly understand the organization:

```text
toxic-comment-analysis/
├── data/
│   ├── test_link.txt                     # Link to the testing dataset in Google Drive
│   ├── train.csv.zip                     # Training dataset (80% of full data)
│   └── test.csv.zip                      # Testing dataset (20% of full data)
│
├── notebooks/
│   ├── descriptive_analysis.ipynb        # Exploratory data analysis on raw/cleaned data
│   ├── svm.ipynb                         # SVM sentiment model
│   ├── nn_toxic_comment_classifier.ipynb # Neural network sentiment model
│   ├── regression.ipynb                  # Logistic regression model
│   ├── ensemble.ipynb                    # Random forest / ensemble model
│   └── test_best_model.ipynb             # Evaluation of the best-performing model
│
├── output/
│   ├── comment-numeric-ensemble-confusionmatrix.png   # Confusion matrix for ensemble model
│   ├── comment-numeric-regression-confusionmatrix.png # Confusion matrix for logistic regression
│   ├── comment-numeric-svm-confusionmatrix.png        # Confusion matrix for SVM model
│   ├── mean_toxicity_by_class.csv                     # Average toxicity score by class
│   ├── numeric-nn-confusionmatrix.png                 # Confusion matrix for neural network model
│   ├── toxicity_distribution_by_class.png             # Toxicity distribution by label
│   ├── word_char_hist.png                             # Word/character count histogram
│   ├── wordcloud_non_toxic.png                        # Word cloud for non-toxic comments
│   └── wordcloud_toxic.png                            # Word cloud for toxic comments
│
├── scripts/
│   ├── data_cleaning.py                # Data cleaning (NA removal, deduplication, encoding)
│   ├── eda_cleaned_data.ipynb          # EDA on cleaned dataset
│   └── eda_raw_data.ipynb              # EDA on raw dataset
│
├── .gitignore                          # Files and folders excluded from version control
├── requirements.txt                    # Project dependencies
└── README.md                           # Project description and documentation
```

## Dataset Description: 
This project uses the Jigsaw Unintended Bias in Toxicity Classification dataset, a large-scale public dataset released on Kaggle. The dataset was originally created to support research on toxic language detection while emphasizing fairness, unintended bias, and the social implications of toxic language models. The link for the dataset is accessible using this link: [Jigsaw Unintended Bias in Toxicity Classification Dataset](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)

The dataset contains:
* A large corpus of user-generated comments from online discussion platforms
* A continuous toxicity score ranging from 0 to 1
* Several auxiliary toxicity indicators, such as:
    * severe_toxicity
    * insult
    * threat
    * obscene
    * identity_attack
* A rich set of identity attributes (e.g., gender, race, religion) used to study model bias

For this project, toxicity labels were derived from the binary rating field:
* rejected → comment flagged or removed due to toxic or inappropriate content
* approved or other statuses → acceptable / non-toxic comment

This rating was used to define the binary classification target:
* Toxic: rating = "rejected"
* Non-toxic: all other cases

## Features Used

Unlike approaches that rely solely on text data, this project incorporated both textual and numeric features:

1: Text Features
* Raw comment text
* Transformed using TF-IDF vectorization (with max_features = 2000)

2: Numeric Features
* Continuous toxicity score
* Auxiliary toxicity indicators (e.g., insult, obscene, identity_attack, etc.)
* These features enriched the model by providing additional context beyond language alone

By combining text-derived embeddings with structured numeric variables, the models were able to learn both linguistic patterns and quantitative signals of toxicity.

After preprocessing (removing nulls, duplicates, and empty comments), the resulting dataset contained approximately 1,593,229 samples, with a naturally imbalanced label distribution in which non-toxic comments are more common than toxic ones.