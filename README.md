# Real News or Fake News? Predicting Headlines Based on Natural Language Processing

## Introduction

This project aims to build a classifier using Natural Language Processing (NLP) techniques to predict whether a news headline is real (truthful) or fake (fabricated). The classifier will take in a dataset of news headlines and generate predictions for unseen data.

The dataset used contains two types of news headlines:

- **Real News (1):** Headlines from credible sources.

- **Fake News (0):** Fabricated or misleading headlines.

The model will use text classification to distinguish between these two types based on their features.

## Objective

1. **Build a Text Classifier:** Use machine learning models (Logistic Regression, Naive Bayes, Random Forest, or Deep Learning models) to classify news headlines.

2. **Text Preprocessing:**
    - Tokenize the headlines.
    - Vectorize the text (using TF-IDF and CountVectorizer).

3. **Training:** Train the model on the `training_data.csv` dataset.

4. **Testing:**
    - Apply the trained model on `testing_data.csv`.
    - Generate a new column in the 'testing_data.csv' file with the predicted labels (0 for fake, 1 for real).

## Installation

To run this project locally, follow these instructions:

### 1. **Clone this repository:**

```bash
git clone https://github.com/martaverfer/project-3-nlp.git 
cd notebooks
```

### 2. **Virtual environment:**

Create the virtual environment: 
```bash
python3 -m venv venv
```

Activate the virtual environment:

- For Windows: 
```bash
venv\Scripts\activate
```

- For Linux/Mac: 
```bash
source venv/bin/activate
```

To switch back to normal terminal usage or activate another virtual environment to work with another project run:
```deactivate```

### 3. **Install dependencies:**

```bash
pip install --upgrade pip; 
pip install -r requirements.txt
```

### 4. **Open the Jupyter notebook to explore the analysis:**

```bash
cd notebooks; 
main.ipynb
```

This script will execute the analysis steps.

## Conclusion
The project applies NLP and machine learning techniques to classify news headlines as real or fake. The model’s goal is to assist in identifying misleading or fabricated information in news sources, which is increasingly important in the age of misinformation.

Good luck, and feel free to reach out if you encounter any issues!
