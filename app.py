# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from urllib.request import urlopen
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import json
import gunicorn
import json
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import sqlite3
import random
from assets import charts


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
with open('./models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)


database_filepath = "./data/DisasterResponse.db"
con = sqlite3.connect(os.path.join(os.path.dirname(__file__), database_filepath))


df2 = None
model_input = ''
df = pd.read_sql_query("SELECT * FROM model_data", con)
category_all = df.columns[2:]


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    },
    'dash-graph2': {
        'height': '700px'
    }
}


app = Dash(__name__)
server = app.server

app.layout = html.Div([

    html.Div([
        html.Div([html.Img(src=app.get_asset_url('logo.png'), height='25 px', width='auto')],
                className = 'col-2', style={'align-items': 'center', 'padding-top' : '1%', 'height' : 'auto'}),
        html.H2('Disaster Response Project'),
        html.P("Analyzing message data for  disaster response."),
        html.Br(),
        html.P("""This ML-model is trained on messages that were send during natural disasters. 
        Classifying these messages, would allow a disaster relief agency to take the appropriate measures. 
        There are 36 pre-defined categories such as "Aid related", "Search and Rescue", "Shelter" or "Medical Help". 
        Use the text input on the left to enter a message for classification."""),
        html.Div([f"The data set consists of {df.shape[0]} samples:"], className='text-padding'),
        # html.Br(),
        html.Div([dcc.Graph(figure=charts.get_pie_chart(df)),], style={'width': '250px', 'align-items': 'center'}),
        html.Div([dcc.Graph(figure=charts.get_category_chart(df))], ),
        ], className='four columns div-user-controls'),


    html.Div([
        html.Div(
        [
        html.Br(),
        html.Br(),
        html.Br(),
                html.H4("Enter a message and hit enter"),

                html.Div(
                    children=[dcc.Input(id="input1", type="text", placeholder="", debounce=True,
                                        style={'border-radius': '8px', #'border': '4px solid red'
                                            'background-color': '#31302f', 'color': 'white',
                                            'width': '100%',
                                                            'padding':'5px'})],  # fill out your Input however you need
                    style=dict(display='flex', justifyContent='center')
                ),
                html.Br(),
                html.Div(id="output1"),
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(f"Categorization based on the {len(category_all)} pre-defined topics"),
        html.Br(),
        html.Div([dcc.Graph(id='result-histogram', figure={}, config={'displayModeBar': False},
                            style={'height': '900px', 'width': '1200px'})], className='dash-graph')
    ], className='eight columns div-for-charts bg-grey')
])


@app.callback(
    Output("output1", "children"),
    Input("input1", "value"),
)
def update_output(input1):
    if not input1:
        return ''
    else:
        model_input = tokenize(input1)
        # print(f"{model_input} from \'{input1}\'")
        txt = ", ".join(str(x) for x in model_input)
        return u'{}'.format(txt)


@app.callback(
    Output('result-histogram', 'figure'),
    Input("output1", "children"))
def update_x_timeseries(output1):
    # use model to predict classification for query
    """ used for testing:
    outp = [random.randint(0,1) for i in category_all]
    outp = [random.choice(category_all) for i in outp if i==1]
    category_all_n = [i.replace("_", " ").title() for i in category_all]
    df2 = pd.DataFrame(data={'cate': category_all_n, 'val': [1 if i in outp else 0 for i in category_all],
                             'color': [str("") for i in category_all]})
    df2['color'] = df2.apply(lambda x: charts.set_c(x['color']), axis=1) """
    classification_labels = model.predict([model_input])[0]
    classification_results = dict(zip(df.columns[2:], classification_labels))
    df_res = pd.DataFrame(data={'cate': [i.replace("_"," ").title() for i in list(classification_results.keys())],
                             'val': classification_results.values()})
    return charts.get_main_chart(df_res)


if __name__ == "__main__":
    app.run_server(debug=True)
