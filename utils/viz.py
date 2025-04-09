# 📚 Basic Libraries
import pandas as pd
import numpy as np

# 📊 Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from wordcloud import WordCloud

# 🤖 Machine Learning
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

col1= '#f2ef12'
col2= '#a4e473'
col3= '#00a181'
col4= '#004651'
palette = [col1, col2, col3]

def imbalance_plot(df, col):
    """
    Visualizes the imbalance of a categorical variable in a DataFrame.
    """
    # Count the occurrences of each category
    counts = df[col].value_counts()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color=[col2, col3])
    plt.title('Class Imbalance')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Fake News (0)', 'Real News (1)'], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('imbalanced.png', transparent=True)
    plt.show()

def wordcloud_plot(text, fake_news = True):
    """
    Visualizes the word cloud of a text column in a DataFrame.
    """
    # Create the wordcloud object
    if fake_news:
        wordcloud = WordCloud(
            width=480, height=480,
            max_font_size=100, min_font_size=10,
            colormap="coolwarm",
            background_color='white'
        ).generate(text)
    else:
        wordcloud = WordCloud(
            width=480, height=480,
            max_font_size=100, min_font_size=10,
            background_color='white'
        ).generate(text)
        
    # Display the generated image:
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)

    if fake_news:
        plt.savefig('fake_wordcloud.png', transparent=True)
    else:
        plt.savefig('real_wordcloud.png', transparent=True)

def metrics_plot(df):
    """
    Plots the classification metrics for different models.
    """
    # Set the model names as the index and drop the 'model' column
    df.set_index('model', inplace=True)

    # Plotting the graph
    plt.figure(figsize=(10, 6))

    # Define the colors for each model
    colors = {
        'Naive Bayes': col1,
        'Random Forest': col2,
        'Logistic Regression': col3
    }

    # Plot each model's metrics as a line
    for model in df.index:
        plt.plot(df.columns, df.loc[model], marker='o', label=model, color=colors[model])

    # Adding labels and title
    plt.title('Model Performance Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Metric Value')
    plt.legend(title='Models')
    plt.grid(True)

    plt.savefig('metrics_lineplot.png', transparent=True)

    # Show the plot
    plt.show()

def confusion_matrix_plot(y_test, y_pred):
    """
    Plots the confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')  
    plt.grid(True)
    # Add a title
    plt.title("Confusion Matrix")

    plt.savefig('confusion_matrix.png', transparent=True)

    plt.show()

def top10_plot(top10, fake =True):
    """
    Plots the top 10 most frequent words for fake news (0).
    """
    if fake:
        color= 'red'
        title= 'Top 10 Most Frequent Words for Fake News (0)'
    else:
        color= col3
        title= 'Top 10 Most Frequent Words for Real News (1)'  

    plt.figure(figsize=(10, 6))
    plt.barh(top10.index, top10.values, color=color)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most frequent word at the top
    if fake:
        plt.savefig('top_10_fake.png', transparent=True)
    else:
         plt.savefig('top_10_real.png', transparent=True)

def vector_plot(accuracy_baseline, accuracy_tfid, bow_accuracy):
    """
    Plots the accuracy of different vectorizers.
    """
    # Create a DataFrame with model names and accuracies
    data = {'Model': ['Baseline', 'TF-IDF', 'BoW'],
            'Accuracy': [accuracy_baseline, accuracy_tfid, bow_accuracy]}

    df = pd.DataFrame(data)

    # Create a bar plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y='Accuracy', data=df, palette=palette)

    # Adding title and labels
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')

    # Show the plot
    plt.ylim([0, 1])  # Set y-axis limits
    plt.savefig('vectors.png', transparent=True)
    plt.show()