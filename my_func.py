# Import needed libraries

import sys
sys.path.append("/Users/patosorio/Turing/Data_Wrangling/Capstone")

import pandas as pd
import numpy as np
import re

# Libraries to visualize data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Library to check correlation between categorical features
from scipy.stats import chi2_contingency

# Machine learning KNN-Neighboor Algorithm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Libraries for choropleth maps
import plotly.express as px #if using plotly
import geopandas as gpd
import pyproj


import missingno as msno


"""
Missingno is a Python library that provides a flexible toolkit 
for visualizing patterns of missing values in datasets. 
The msno.matrix function creates a matrix visualization of the missing 
values in the dataset. 
White lines indicate missing values. 
This visualization can help us quickly understand the patterns of 
missingness in the dataset, including the amount and location of missing data, 
and any correlations between missing values in different columns.
"""

# Params settings

sns.set()
plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['figure.dpi'] = 80

import warnings
warnings.filterwarnings('ignore')

# Matrix function with plot edits

def missing_matrix(df, **kwargs):
    # Call the original matrix function
    msno.matrix(df, **kwargs)

    # Access the Axes object from the current plot
    ax = plt.gca()

    # Wrap text on x-axis tick labels
    ax.set_xticklabels(ax.get_xticklabels(), wrap=True)

    # Set smaller font size for x-axis tick labels
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)



# Function that replaces NaN values of the city column based on the most common city 
# within the same province, country, and state

def fill_missing_value(row, df, target_column, conditions):
    if pd.isna(row[target_column]):
        filter_condition = True
        for condition in conditions:
            filter_condition &= (df[condition] == row[condition])

        most_common_value = df.loc[filter_condition, target_column].mode()
        
        if len(most_common_value) > 0:
            return most_common_value[0]
        else:
            return 'Unknown'
    else:
        return row[target_column]

    
# Function to apply to columns that change to float all numbers that are 
# strings with less than the limit characters

def convert_to_float(x, limit: int):
    if isinstance(x, str):
        if len(x) <= limit:
            return float(x)
        else:
            return np.nan
    return x


# Function to check categorical correlations between a target column and a list of other columns.

def chi2_correlation(target_column: str, categorical_columns: list, df):

    # Calculate chi-squared test for categorical features
    for col in categorical_columns: 
        contingency_table = pd.crosstab(df[col], df[target_column], dropna=True)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results = print(f"{col} association with {target_column} - chi2: {chi2}, p-value: {p_value}")
        
    # Print the p_value   
    return results



# PLOTS FUNCTIONS : 


def counterplot(target_column: int, df, xlabel_string: str):
    # Create countplot from arguments
    sns.countplot(x=target_column, data=df, palette='Set2')
    # Set the size of the figure

    # Set plot labels
    plt.xlabel(xlabel_string.capitalize())
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    
    # Show plot
    plot = plt.show()
    
    return plot

def counterplot_hue(df, target_column: int, hue: str, xlabel:str, title: str):
    # Create a count plot with 'Age' on the x-axis and separate bars for 'Sex'
    sns.countplot(x=target_column, hue=hue, data=df, palette='Set2')

    # Set plot labels
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=90)

    # Show plot
    plt.show()



def histogram(df, column, bins=50):
    # Sort the values in the column
    sorted_values = df[column].sort_values()

    # Plot a histogram of the sorted values
    sns.histplot(sorted_values, bins=bins, color=sns.color_palette("Set2")[0])

    # Set plot labels
    plt.xlabel(column)
    plt.ylabel("Count")

    # Show plot
    plt.show()
    
    
    
def pie_chart(df, column):
    # Get the counts of each category in the column
    counts = df[column].value_counts()

    # Create a color palette
    palette = sns.color_palette("Set2", len(counts))

    # Create a pie chart of the counts
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=palette)

    # Add a title
    plt.title(f'Distribution of {column}')

    # Show the plot
    plt.show()
    
    
def barplot(column, hue, df, title):
    
    # Create the bar plot
    sns.barplot(x=column, y=hue, data=df, palette='viridis')

    # Set plot labels and title
    plt.xlabel(column)
    plt.ylabel(hue)
    plt.title(title)
    plt.xticks(rotation=90)

    plt.show()



def heatmap(df, index, columns, values):
    # Create a pivot table from the DataFrame
    pivot_table = df.pivot_table(index=index, columns=columns, 
                                 values=values, aggfunc='size', fill_value=0)

    # Create a heatmap from the pivot table
    sns.heatmap(pivot_table, cmap="Set2", annot=True, fmt="d")

    # Show the plot
    plt.show()



def create_map(column: str, shp_path: str, index_name:str, title: str, df):
    
    #set up the file path and read the shapefile
    shapefile = gpd.read_file(shp_path)

    merged = shapefile.set_index('name_1').join(df.set_index(column))

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axis('off')
    ax.set_title(title, fontdict={'fontsize': '15', 'fontweight' : '3'})
    merged.plot(column=index_name,
                #cmap=set2_cmap,
                linewidth=0.9,
                ax=ax,
                edgecolor='1',
                legend=True, missing_kwds={
                "color": "lightgrey",
                "label": "Missing values",},)
    
    plt.show()


def time_series(df, x: str, y: str, title: str, y_label):
    
    sns.lineplot(data=df, x=x, y=y,color=sns.color_palette("Set2")[0])
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Date')
    
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=90)
    
    plt.show()



# Define categories and corresponding keywords/patterns
categories = {
    'Overseas Inflow': ['overseas inflow'],
    'Contact with Patient': ['contact with patient'],
    'Shopping & Food & Entertainment Venue': ['itaewon', 'clubs', 'gu', 'center','richway', 'yangcheon', 'table',
                                             'silver', 'shelter', 'collective', 'karaoke'],
    'Workplace': ['guro', 'logistics', 'coupang', 'facility', 'factory', 'company', 'workplace', 
                  'office', 'manufacture', 'manufacturing', 'samsung', 'electronics', 'guri', 
                  'collective', 'business', 'industrial', 'industry', 'construction', 'enterprise',
                  'fishing', 'fishery', 'fisheries', 'machine', 'brothers', 'call', 'shincheonji', 'coupang',
                 'meeting', 'worker', 'group', 'planted'],
    'Church': ['church', 'community', 'churches', 'christ','biblical', 'pastors', 'pilgrimage', 'israel', 'jeil'],
    'Gym Facility': ['gym'],
    'Admin Centers': ['ministry', 'insurance', 'news', 'fire'],
    'Education': ['school', 'university', 'learning', 'class', 'students', 'language', 'study','infection' ],
    'Hospital': ['hospital', 'medical', 'clinic', 'healthcare', 'doctor', 'nurse', 'nursing', 'home', 'care'],
    'Unknown': ['Unknown']
}

# Function to categorize a case based on its description
def categorize_case(description):
    for category, keywords in categories.items():
        for keyword in keywords:
            if re.search(keyword, description, re.IGNORECASE):
                return category
    return 'Other'  # Default category if no match is found 
