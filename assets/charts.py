# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import random
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Define color sets of paintings
night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)']


def set_c(color):
    return random.choice(px.colors.qualitative.Set2)

def get_pie_chart(df):
    #df3 = pd.DataFrame(data={'color': [str("") for i in category_all]})
    #df3['color'] = df3.apply(lambda x: set_c(x['color']), axis=1)
    fig = go.Figure(data=[go.Pie(values=df.genre.value_counts().to_list(), labels=df.genre.unique(), textinfo='label',
                                      insidetextorientation='radial', hole=.25, marker_colors=night_colors)])
    fig.update_layout(legend_font_size=14, font=dict(family="Open Sans"))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update(layout_showlegend=False)
    fig.update_traces(marker=dict(line=dict(color='#9c9c9c', width=1.5)))
    fig.update_traces(opacity=0.7)
    return fig


def get_category_chart(df):
    # Plotting of Categories Distribution in Direct Genre
    direct_cate = df[df.genre == 'direct']

    direct_cate_counts = (direct_cate.mean() * direct_cate.shape[0]).sort_values(ascending=False)
    direct_cate_names = list(direct_cate_counts.index)
    fig = px.bar(x=[i.replace("_", " ").title() for i in direct_cate_names], y=direct_cate_counts)
    fig.update_traces(marker_color=night_colors[0], marker_line_color='#9c9c9c', marker_line_width=1, opacity=0.7)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.update_layout(xaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='#9c9c9c'),
                     title_font_color='#9c9c9c', mirror=True,
                     ticks='outside', showline=True, gridwidth=1, gridcolor='#4c4c4c')
    fig.update_xaxes(tickfont=dict(family='Helvetica', color='#9c9c9c'),
                     title_font_color='#9c9c9c', mirror=True,
                     ticks='outside', showline=True, gridwidth=1, gridcolor='#4c4c4c')
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    return fig


def get_main_chart(df):
    fig = px.bar(df, x='val', y='cate', hover_data=['cate'], orientation='h')#, color='color')
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_yaxes(tickfont=dict(family='Helvetica', color='#9c9c9c'),
                     title_font_color='#9c9c9c',
                     ticks='outside', showline=True, gridwidth=1, gridcolor='#4c4c4c')
    fig.update_layout(yaxis_title=None)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(marker_color=night_colors[0], marker_line_color='#9c9c9c', marker_line_width=1, opacity=0.7)
    return fig
