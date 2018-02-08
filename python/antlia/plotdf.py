# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def colormap(dataframe, color_key, color_palette):
    """Return a vector of colors for each dataframe column, to be used with
    seaborn plots.

    Parameters:
    dataframe: pandas dataframe
    color_key: column in 'dataframe' to use for an index in 'color_palette'
    color_palette: nx3 color array
    """
    msg = 'column \'{}\' not in dataframe'.format(color_key)
    assert color_key in dataframe.columns.values, msg

    x = np.vectorize(lambda x: color_palette[int(x)])(
            dataframe.as_matrix([color_key]))
    return np.array(x).transpose().reshape((-1, 3))

def plotjoint(x, y, dataframe, kde_key=None, color_map=None, g=None, **kwargs):
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
               plot. If 'kde_key' is not None, this parameter is ignored.
    """
    if g is None:
        # initialize figure
        g = sns.JointGrid(x=x, y=y, data=dataframe)
    else:
        # update x and y data
        g.x = np.asarray(dataframe.get(x, x))
        g.y = np.asarray(dataframe.get(y, y))

    # draw scatter plot on figure
    if kde_key is None and color_map is None:
        g.plot_joint(plt.scatter, **kwargs)
    else:
        if kde_key is not None:
            color_map = colormap(dataframe, *kde_key)
        g.plot_joint(plt.scatter, color=color_map, **kwargs)

    if kde_key:
        color_key, color_palette = kde_key
        with warnings.catch_warnings():
            # suppress warnings from plotting multiple kde plots
            warnings.simplefilter('ignore', RuntimeWarning)
            for r in dataframe[color_key].unique():
                sns.kdeplot(dataframe[x][dataframe[color_key] == r],
                            ax=g.ax_marg_x, vertical=False,
                            color=color_palette[int(r)], shade=True)
                sns.kdeplot(dataframe[y][dataframe[color_key] == r],
                            ax=g.ax_marg_y, vertical=True,
                            color=color_palette[int(r)], shade=True)
    else:
        # use color from scatter plot
        g.plot_marginals(sns.kdeplot, shade=True)

    try:
        g.ax_marg_x.legend_.remove()
        g.ax_marg_y.legend_.remove()
    except AttributeError:
        pass

    # rescale limits of kde plots
    g.ax_marg_x.relim()
    g.ax_marg_x.autoscale()
    g.ax_marg_y.relim()
    g.ax_marg_y.autoscale()

    return g
