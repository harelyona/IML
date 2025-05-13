# Basic imports and settings for working with data
import numpy as np
import pandas as pd

# Imports and settings for plotting of graphs
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Tuple


pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=20, r=20, t=40, b=0)
    )
)
pio.templates.default = "simple_white+custom"


custom = [[0.0, "rgb(165,0,38)"],
          [0.1111111111111111, "rgb(215,48,39)"],
          [0.2222222222222222, "rgb(244,109,67)"],
          [0.3333333333333333, "rgb(253,174,97)"],
          [0.4444444444444444, "rgb(254,224,144)"],
          [0.5555555555555556, "rgb(224,243,248)"],
          [0.6666666666666666, "rgb(171,217,233)"],
          [0.7777777777777778, "rgb(116,173,209)"],
          [0.8888888888888888, "rgb(69,117,180)"],
          [1.0, "rgb(49,54,149)"]]

class_symbols = np.array(["circle", "x", "diamond"])
class_colors = lambda n: [custom[i] for i in np.linspace(0, len(custom)-1, n).astype(int)]




def decision_surface(predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()])

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers", marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)



def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75, seed: int = 0) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    train = X.sample(frac=train_proportion, random_state=seed)
    test = X.loc[X.index.difference(train.index)]
    return train, y.loc[train.index], test, y.loc[test.index]
