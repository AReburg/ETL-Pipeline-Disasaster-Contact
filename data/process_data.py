import sys
import os
import pandas as pd
import numpy as np
import sqlite3
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# download necessary NLTK data
# nltk.download('omw-1.4')
# nltk.download(['punkt', 'wordnet'])
#from sqlalchemy import create_engine
# from pathlib import Path
# cwd = Path(__file__).parent
# dirname = cwd.parent / "data"


def load_data(messages_filepath, categories_filepath):
    """ load text messages as well as the category and merge both dataframes """

    df_message = pd.read_csv(os.path.join(os.path.dirname(__file__), 'disaster_messages.csv'), sep=',',
                             index_col=False, encoding='utf-8')
    df_category = pd.read_csv(os.path.join(os.path.dirname(__file__), 'disaster_categories.csv'), sep=',',
                              index_col=False, encoding='utf-8')

    df_message.drop_duplicates(subset="id", keep=False, inplace=True)
    df_category.drop_duplicates(subset="id", keep=False, inplace=True)
    df = df_category.merge(df_message, how='left', on=['id'])

    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # rename columns with striped col name -> remove last two characters ("-1" or "-0")
    categories.columns = row.apply(lambda x: x[:-2])
    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.replace(2, 1, inplace=True)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    return df


def categorize(categories_text):
    """ """
    categories = []
    for i in categories_text.split(";"):
        if i.find("-1") > 0:
             categories.append(i)
    categories = ",".join(str(x).strip('-1').replace("_", " ").title() for x in categories)
    return categories


def clean_data(df):
    """ clean raw data """
    df = df.drop(['id', 'original'], axis=1)
    #df['categories'] = df.apply(lambda x: categorize(x.categories), axis=1)
    #df['categories'].replace('', np.nan, inplace=True)
    #df.dropna(subset=['categories'], inplace=True)
    # print(df.shape[0]) # -> there are categories rows with ""
    return df


def save_data(df, database_filename):
    """save datafram as .db file """
    con = sqlite3.connect(os.path.join(os.path.dirname(__file__), database_filename))
    df.to_sql(name='model_data', if_exists='replace', index=False, con=con)


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()