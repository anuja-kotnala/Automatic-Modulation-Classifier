import matplotlib.pyplot as plt

def setup_publication_style():
    """
    Sets up publication-quality matplotlib parameters.
    Ensures clear grids, readable labels, and modern color schemes.
    """
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'grid.color': '#cccccc',
        'axes.grid': True,
        'axes.facecolor': '#fafafa',
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
