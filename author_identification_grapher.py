from flask import render_template

import re
import json
import plotly

import pandas as pd
import numpy as np
from urllib.parse import quote


def make_graph(input1, result):

    graphs = []




    graph_title = "Authors"
    graph = dict(
        data=[
            dict(
                x=[r[0] for r in result],
                y=[r[1] for r in result],
                type='bar',
                name="score"
            )
        ],
        layout=dict(
            title=graph_title,
            orientation='h',
            xaxis=dict(
                title='author',
                range=[-0.5, len(result) + 0.5]
            ),
            yaxis=dict(
                title='probability'
            )
        )
    )
    
    graphs.append(graph)
    ids = [g["layout"]["title"] for g in graphs]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    print ("num graphs:", len(graphs), graphJSON, ids)


    return render_template('layouts/index.html',
                           ids=ids,
                           input1=input1,
                           output=str(result),
                           graphJSON=graphJSON)
