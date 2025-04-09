# 📚 Basic Libraries
import pandas as pd
import os
import pickle

# 📝 Text Processing
import nltk
from nltk.corpus import wordnet # to get the wordnet pos tags

# 🤖 Machine Learning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def map_pos_tag(word):
    """
    Map POS tag to first character lemmatize() accepts.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper() # get the first character of the POS tag
    tag_dict = { 
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN) # return the value of the key or the default value

def save_dataframe_to_pickle(df, filename):
    """
    Saves the DataFrame to a pickle file if the file doesn't already exist.
    """
    if not os.path.exists(filename):  # Check if the file already exists
        df.to_pickle(filename)  # Save DataFrame as pickle file
        print(f"DataFrame saved as {filename}")
    else:
        print(f"{filename} already exists. File not overwritten.")

def classification_metrics(X_train, X_test, y_train, y_test, models):
    """
    Returns a dataframe with the metrics of the models you have passed as a dictionary.
    """
    # List to store results for each test size
    results = []

    for model_name, model in models.items():
       
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, predictions) 
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Store the results for this test size
        results.append({
            'model': model_name,
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3)
        })

    pd.set_option('display.float_format', '{:.3f}'.format)

    # Return the final dataframe containing all results
    return pd.DataFrame(results)

def save_model_to_pickle(model, filename):
    """
    Saves the classification model to a pickle file in the /models folder if the file doesn't already exist.
    """
    # Define the path to the models folder
    models_folder = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Ensure the models folder exists
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    # Define the full path for the pickle file
    full_path = os.path.join(models_folder, filename)
    
    # Check if the file already exists
    if not os.path.exists(full_path):
        # Save the model as a pickle file
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved as {full_path}")

    else:
        print(f"{full_path} already exists. File not overwritten.")

def save_dataframe_to_csv(df, filepath):
    """
    Saves the dataframe to a CSV file in the /data folder if the file doesn't already exist.
    """
    # Check if the file already exists
    if not os.path.exists(filepath):
        # If the file doesn't exist, save the dataframe
        df.to_csv(filepath, index=False)
        print(f"File saved as {filepath}")
    else:
        print(f"The file {filepath} already exists. Skipping save.")