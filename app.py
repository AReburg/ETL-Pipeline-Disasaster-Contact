# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import re
import json
import gunicorn
import os
import pickle
import sqlite3
import nltk
from assets import charts
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import multioutput
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import f1_score, classification_report, make_scorer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

# load db file
database_filepath = "./data/DisasterResponse.db"
con = sqlite3.connect(os.path.join(os.path.dirname(__file__), database_filepath))
df = pd.read_sql_query("SELECT * FROM model_data", con)


def tokenize(text):
    """ function to display the tokenized input needed for building the model """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load model from .pkl file
with open('./models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)


# set-up webpage layout to display cool visuals and receives user input text for model
# render web page with plotly graphs
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    # left half of the web page
    html.Div([
        html.Div([html.Img(src=app.get_asset_url('logo.png'), height='25 px', width='auto')],
                className = 'col-2', style={'align-items': 'center', 'padding-top' : '1%', 'height' : 'auto'}),
        html.H2('Disaster Response Project'),
        html.P("Analyzing message data for  disaster response."),
        html.Br(),
        html.P("""This ML-model is trained on messages that were send during natural disasters. 
        Classifying these messages, would allow a disaster relief agency to take the appropriate measures. 
        There are 36 pre-defined categories such as "Aid related", "Search and Rescue", "Shelter" or "Medical Help". 
        Use the text input on the right to enter a message for classification."""),
        html.Div([f"The data set consists of {df.shape[0]} samples:"], className='text-padding'),
        html.Div([dcc.Graph(figure=charts.get_pie_chart(df), config={'displayModeBar': False})], style={'width': '250px', 'align-items': 'center'}),
        html.Div([dcc.Graph(figure=charts.get_category_chart(df), config={'displayModeBar': False})]),
        ], className='four columns div-user-controls'),

    # right half of the web page
    html.Div([
        html.Div(
            [
            html.Br(),
            html.Br(),
            html.Br(),
                html.H4("Enter a message and hit enter"),

                html.Div(
                    children=[dcc.Input(id="input_text", type="text", placeholder="", debounce=True,
                                        style={'border-radius': '8px', #'border': '4px solid red'
                                            'background-color': '#31302f', 'color': 'white',
                                            'width': '100%',
                                                            'padding':'5px'})],  # fill out your Input however you need
                    style=dict(display='flex', justifyContent='center')
                ),
                html.Br(),
                html.Div(id="tokenized_text"),
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(f"Categorization based on the {len(df.columns[2:])} pre-defined topics"),
        html.Br(),
        html.Div([dcc.Graph(id='result-histogram', figure={}, config={'displayModeBar': False},
                            style={'height': '900px', 'width': '1200px'})], className='dash-graph')
    ], className='eight columns div-for-charts bg-grey')
])


@app.callback(
    [Output('result-histogram', 'figure'), Output('tokenized_text', 'children')],
    Input("input_text", "value"))
def update_categories(input_text):
    """ use model to predict classification for input text query """
    if input_text == '' or input_text is None:
        df_res = pd.DataFrame(data={'cate': [i.replace("_", " ").title() for i in df.columns[2:]],
                                    'val': [0 for _ in df.columns[2:]]})
        tokenized_text = ""

    else:
        classification_labels = model.predict([input_text])[0]
        classification_results = dict(zip(df.columns[2:], classification_labels))
        df_res = pd.DataFrame(data={'cate': [i.replace("_"," ").title() for i in list(classification_results.keys())],
                                 'val': classification_results.values()})
        tokenized_text = ", ".join(str(x) for x in tokenize(input_text))

    return [charts.get_main_chart(df_res), tokenized_text]


if __name__ == "__main__":
    app.run_server(debug=False)

