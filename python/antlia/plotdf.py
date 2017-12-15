# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def colormap(dataframe, color_key, color_palette):
    """Return a vector of colors for each dataframe column, to be used with
    seaborn plots.

    Parameters:
    dataframe: pandas dataframe
    color_key: column in 'dataframe' to use for an index in 'color_palette'
    color_palette: nx3 color array
    """
    x = np.vectorize(lambda x: color_palette[int(x)])(
            dataframe.as_matrix([color_key]))
    return np.array(x).transpose().reshape((-1, 3))

def plotjoint(x, y, dataframe, kde_key=None, color_map=None):
    """Return a vector of colors for each dataframe column, to be used with
    seaborn plots.

    Parameters:
    x: string, column in 'dataframe'
    y: string, column in 'dataframe'
    dataframe: pandas dataframe
    kde_key: (color_key, color_palette), If not 'None', draw a kdeplot per
             element in 'color_key' with color determined by 'color_palette'.
             This will also set the color of elements in the scatter plot.
    color_map: nx3 color array, If not 'None' and 'kde_key' is not None, plot
               each element with color specified by corresponding row in scatter
               plot. If this is 'kde_key' is not None, this parameter is
               ignored.
    """
    g = sns.JointGrid(x=x, y=y, data=dataframe)
    if kde_key is None and color_map is None:
        g.plot_joint(plt.scatter)
    else:
        if kde_key is not None:
            color_map = colormap(dataframe, *kde_key)
        g.plot_joint(plt.scatter, color=color_map)

    if kde_key:
        color_key, color_palette = kde_key
        for r in dataframe[color_key].unique():
            sns.kdeplot(dataframe[x][dataframe[color_key] == r],
                        ax=g.ax_marg_x, vertical=False,
                        color=color_palette[int(r)], shade=True)
            sns.kdeplot(dataframe[y][dataframe[color_key] == r],
                        ax=g.ax_marg_y, vertical=True,
                        color=color_palette[int(r)], shade=True)
    else:
        g.plot_marginals(sns.kdeplot, color='black', shade=True)

    g.ax_marg_x.legend_.remove()
    g.ax_marg_y.legend_.remove()
    return g
