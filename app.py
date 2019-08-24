import base64
import logging
import os
import re

import advertools as adv
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go

from urllib.parse import quote
from dash_table import DataTable
from dash_table.FormatTemplate import Format
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

logging.basicConfig(level=logging.INFO)

img_base64 = base64.b64encode(open('./logo.png', 'rb').read()).decode('ascii')

adv.SERP_GOOG_VALID_VALS['searchType'] = {'image', 'web'}

docs_df = pd.read_csv('api_docs_df.csv')
docs_params = {k: v for k, v in zip(docs_df['Parameter name'],
                                    docs_df['Description'])}
del docs_params['googlehost']
cx = os.environ['GOOG_CSE_CX']
key = os.environ['GOOG_CSE_KEY']

app = dash.Dash(__name__,  external_stylesheets=[dbc.themes.COSMO])

server = app.server

app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Google Search Results Rankings (SERPs)
            {%title%}board | advertools </title>
            {%favicon%}
            {%css%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%} 
            </footer>
        </body>
    </html>
"""

app.layout = html.Div([
    html.Br(),
    dbc.Row([
       dbc.Col([
           html.Img(src='data:image/png;base64,' + img_base64, width=200),
           html.A(['online marketing', html.Br(), 'productivity & analysis'],
                  href='https://github.com/eliasdabbas/advertools')
       ], sm=12, lg=2, style={'text-align': 'center'}), html.Br(),
       dbc.Col([
           html.H2('Google Search Results Pages Rankings',
                   style={'text-align': 'center'}),
       ], sm=12, lg=9),
    ], style={'margin-left': '1%'}),

    dbc.Row([

        dbc.Col([
            html.Br(),
            html.Div(dbc.Button(id='button', children='Submit', size='lg'),
                     style={'text-align': 'center'}), html.Br(),
            dbc.Textarea(id='query', rows=4,
                         placeholder='Enter keywords\none\nper\nline'),
        ] + [
            dcc.Dropdown(id=key,
                         placeholder=re.sub('[A-Z]',  r' \g<0>', key).lower(),
                         multi=True,
                         options=[{'label': x, 'value': x} for x in
                                  sorted(adv.SERP_GOOG_VALID_VALS.get(key))])
            if adv.SERP_GOOG_VALID_VALS.get(key) else
            dbc.Input(id=key, placeholder=re.sub('[A-Z]',
                                                 r' \g<0>', key).lower())
            for key in docs_params
        ] + [
            dbc.Tooltip(docs_params[key][:450], target=key)
            for key in docs_params
        ] + [html.Br() for i in range(6)],
                lg=2, xs=12,style={'margin-left': '1%', 'align': 'center'}),
        dbc.Col([
            dcc.Loading([
                dcc.Store('serp_results'),
            ], type='circle'),
            dbc.Row([
                dbc.Col([
                    dbc.Label('Number of domains to plot:'),
                    dcc.Dropdown(id='num_domains_to_plot',
                                 options=[{'label': num, 'value': num}
                                          for num in range(1, 51)],
                                 value=10),
                ], lg=3, xs=8),
                dbc.Col([
                    dbc.Label('Compare specific domains:'),
                    dcc.Dropdown(id='select_domain',
                                 multi=True,
                                 options=[{'label': n, 'value': n}
                                          for n in range(1, 11)]),
                    dbc.Container(id='opacity_output')
                ], lg=3, xs=8),
            ], id='chart_controls', style={'display': 'none'}),
            dcc.Loading([
                dcc.Graph(id='serp_graph',
                          figure={'layout': {'paper_bgcolor': '#eeeeee',
                                             'plot_bgcolor': '#eeeeee',
                                             'yaxis': {'zeroline': False},
                                             'xaxis': {'zeroline': False}}},
                          config={'displayModeBar': False}),
            ]),
            html.Br(), html.Br(),
            html.Div(html.A('Download Table', id='download_link',
                            download="rawdata.csv", href="", target="_blank",
                            n_clicks=0), style={'text-align': 'right'}),
            dcc.Loading(
                DataTable(id='serp_table',
                          fixed_rows={'headers': True},
                          sort_action='native',
                          style_cell_conditional=[{
                              'if': {'row_index': 'odd'},
                              'backgroundColor': '#eeeeee'}],
                          style_cell={'font-family': 'Source Sans Pro',
                                      'width': '130px'},
                          virtualization=True)
            ),
        ], lg=9, xs=11)
    ]),
], style={'background-color': '#eeeeee'})


@app.callback(Output('select_domain', 'options'),
              [Input('serp_results', 'data')])
def set_domain_select_options(data_df):
    df = pd.DataFrame(data_df)
    domains = df['displayLink'].unique()
    return [{'label': domain.replace('www.', ''), 'value': domain}
            for domain in domains]


@app.callback(Output('download_link', 'href'),
              [Input('serp_results', 'data')])
def download_df(data_df):
    df = pd.DataFrame(data_df)
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + quote(csv_string)
    log_msg = (format(df.memory_usage().sum(), ',') +
               'bytes, shape:' + str(df.shape))
    logging.info(msg=log_msg)
    return csv_string


@app.callback(Output('serp_results', 'data'),
              [Input('button', 'n_clicks')],
              [State('query', 'value')] +
              [State(key, 'value')for key in docs_params])
def get_serp_data_save_to_store(n_clicks, query, *args):
    if query is None:
        raise PreventUpdate
    kwargs = {k: arg for k, arg in zip(docs_params.keys(), args)}
    if kwargs['searchType'] and 'web' in kwargs['searchType']:
        kwargs['searchType'][kwargs['searchType'].index('web')] = None
    adv.SERP_GOOG_VALID_VALS['searchType'] = {'image', None}
    q = [word.strip() for word in query.split('\n')]
    log_msg = {'q': q, **kwargs}
    logging.info(msg=log_msg)
    df = adv.serp_goog(cx=cx, key=key, q=q, **kwargs)

    return df.to_dict('rows')


@app.callback([Output('serp_table', 'data'),
               Output('serp_table', 'columns')],
              [Input('serp_results', 'data')])
def populate_table_data(serp_results):
    if serp_results is None:
        raise PreventUpdate
    df = pd.DataFrame(serp_results, columns=serp_results[0].keys())
    del df['pagemap']
    columns = [{'name': re.sub('[A-Z]', r' \g<0>', x).title(), 'id': x,
                'format': Format(group=',') if 'esults' in x else None}
               for x in df.columns]
    last_col = df.columns.tolist().index('totalResults') + 1
    return df.iloc[:, :last_col].to_dict('rows'), columns[:last_col]


@app.callback([Output('serp_graph', 'figure'),
               Output('chart_controls', 'style')],
              [Input('serp_results', 'data'),
               Input('num_domains_to_plot', 'value'),
               Input('select_domain', 'value')])
def plot_data(serp_results, num_domains=10, select_domain=None):
    if serp_results is None:
        raise PreventUpdate
    df = pd.DataFrame(serp_results, columns=serp_results[0].keys())
    if select_domain:
        df = df[df['displayLink'].isin(select_domain)]
    top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()
    top_df = df[df['displayLink'].isin(top_domains)]
    top_df_counts_means = (top_df
                           .groupby('displayLink', as_index=False)
                           .agg({'rank': ['count', 'mean']}))
    top_df_counts_means.columns = ['displayLink', 'rank_count', 'rank_mean']
    top_df = (pd.merge(top_df, top_df_counts_means)
              .sort_values(['rank_count', 'rank_mean'],
                           ascending=[False, True]))
    rank_counts = (top_df
                   .groupby(['displayLink', 'rank'])
                   .agg({'rank': ['count']})
                   .reset_index())
    rank_counts.columns = ['displayLink', 'rank', 'count']
    summary = (df
               .groupby(['displayLink'], as_index=False)
               .agg({'rank': ['count', 'mean']})
               .sort_values(('rank', 'count'), ascending=False)
               .assign(coverage=lambda df: (df[('rank', 'count')]
                                            .div(df[('rank', 'count')]
                                                 .sum()))))
    summary.columns = ['displayLink', 'count', 'avg_rank', 'coverage']
    summary['displayLink'] = summary['displayLink'].str.replace('www.', '')
    summary['avg_rank'] = summary['avg_rank'].round(1)
    summary['coverage'] = (summary['coverage'].mul(100)
                           .round(1).astype(str).add('%'))
    num_queries = df['queryTime'].nunique()

    fig = go.Figure()
    fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),
                    y=top_df['rank'], mode='markers',
                    marker={'size': 30, 'opacity': 1/rank_counts['count'].max()})

    fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),
                    y=rank_counts['rank'], mode='text',
                    text=rank_counts['count'])

    for domain in rank_counts['displayLink'].unique():
        rank_counts_subset = rank_counts[rank_counts['displayLink'] == domain]
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[0], mode='text',
                        marker={'size': 50},
                        text=str(rank_counts_subset['count'].sum()))

        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[-1], mode='text',
                        text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[-2], mode='text',
                        marker={'size': 50},
                        text=str(round(rank_counts_subset['rank']
                                       .mul(rank_counts_subset['count'])
                                       .sum() / rank_counts_subset['count']
                                       .sum(), 2)))

    minrank, maxrank = min(top_df['rank'].unique()), max(top_df['rank'].unique())
    fig.layout.yaxis.tickvals = [-2, -1, 0] + list(range(minrank, maxrank+1))
    fig.layout.yaxis.ticktext = ['Avg. Pos.', 'Coverage', 'Total<br>appearances'] + list(range(minrank, maxrank+1))

    fig.layout.height = max([600, 100 + ((maxrank - minrank) * 50)])
    fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = '#eeeeee'
    fig.layout.plot_bgcolor = '#eeeeee'
    fig.layout.autosize = False
    fig.layout.margin.r = 2
    fig.layout.margin.l = 120
    fig.layout.margin.pad = 0
    fig.layout.hovermode = False
    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.zeroline = False
    return [fig, {'display': 'inline-flex', 'width': '100%'}]


if __name__ == '__main__':
    app.run_server()
