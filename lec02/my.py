import matplotlib.pyplot as plt
import numpy as np

def output_svg():
    # set figure size
    plt.rcParams["figure.figsize"] = (8,4)
    # produce vector inline graphics
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('svg')

def plot_decision_regions(X, Y, clf, ax=None, N=200):
    """Returns the plot of decision boundaries
    
    Parameters:
    X (pandas DataFrame): 2 input features
    Y (pandas Series):  class
    clf (sklean Classifier): A classifier trained to predicts Y from X
    ax (axis): axis to plot the boundaries
    N: number of points for each dimension to scan for the decision boundaries
    
    Return:
    axis: plot the decision boundaries
    """
    if ax is None:
        ax = plt.gca()

    X_min = X.min()
    X_max = X.max()
    x1, x2 = np.meshgrid(np.linspace(X_min[0], X_max[0], N),
                         np.linspace(X_min[1], X_max[1], N))
    rename = dict(zip(Y.cat.categories,range(len(Y.cat.categories))))
    
    y = [rename[v] for v in Y]
    yhat = np.array([rename[v] for v in clf.predict(np.c_[x1.ravel(), x2.ravel()])])

    z = yhat.reshape(x1.shape)

    ax.contourf(x1,x2,z,alpha=0.4)
    ax.scatter(X.iloc[:,0],X.iloc[:,1],c=y,edgecolor='w',s=20)
    ax.set_xlim(X_min[0],X_max[0])
    ax.set_ylim(X_min[1],X_max[1])
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])

    return ax