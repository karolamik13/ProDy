"""This module defines utility functions for drawing."""
# Note for developers: this module should be imported only a specific function is needed. 
# Please do not import this module when initializing ProDy for robustness reasons.

import numpy as np

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

__all__ = ['Arrow3D', 'drawArrow3D', 'drawTree']

class Arrow3D(FancyArrowPatch):
    """This function is implemented by tacaswell on stackoverflow: 
    https://stackoverflow.com/a/29188796."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super(Arrow3D, self).__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def drawArrow3D(*xyz, **kwargs):
    from matplotlib.pyplot import gca

    allargs = xyz
    xs = []; ys = []; zs = []; args = []

    for arg in allargs:
        if not np.isscalar(arg):
            if len(arg) == 3:
                xs.append(arg[0])
                ys.append(arg[1])
                zs.append(arg[2])
                continue
        args.append(arg)

    ax = gca()
    if 'arrowstyle' not in kwargs:
        kwargs['arrowstyle'] = '-|>'
    if 'mutation_scale' not in kwargs:
        kwargs['mutation_scale'] = 20
    if 'color' not in kwargs:
        kwargs['color'] = 'k'

    arrow = Arrow3D(xs, ys, zs, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow

def drawTree(
    tree,
    label_func=str,
    show_confidence=True,
    # For power users
    orientation='horizontal',
    inverted=False,
    branch_labels=None,
    label_colors=None,
    *args,
    **kwargs
):
    """Plot the given tree using matplotlib (or pylab).

    The graphic is a rooted tree, drawn with roughly the same algorithm as
    draw_ascii.

    Additional keyword arguments passed into this function are used as pyplot
    options. The input format should be in the form of:
    pyplot_option_name=(tuple), pyplot_option_name=(tuple, dict), or
    pyplot_option_name=(dict).

    Example using the pyplot options 'axhspan' and 'axvline'::

        from Bio import Phylo, AlignIO
        from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
        constructor = DistanceTreeConstructor()
        aln = AlignIO.read(open('TreeConstruction/msa.phy'), 'phylip')
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(aln)
        tree = constructor.upgma(dm)
        Phylo.draw(tree, axhspan=((0.25, 7.75), {'facecolor':'0.5'}),
        ... axvline={'x':0, 'ymin':0, 'ymax':1})

    Visual aspects of the plot can also be modified using pyplot's own functions
    and objects (via pylab or matplotlib). In particular, the pyplot.rcParams
    object can be used to scale the font size (rcParams["font.size"]) and line
    width (rcParams["lines.linewidth"]).

    :Parameters:
        label_func : callable
            A function to extract a label from a node. By default this is str(),
            but you can use a different function to select another string
            associated with each node. If this function returns None for a node,
            no label will be shown for that node.
        show_confidence : bool
            Whether to display confidence values, if present on the tree.
        branch_labels : dict or callable
            A mapping of each clade to the label that will be shown along the
            branch leading to it. By default this is the confidence value(s) of
            the clade, taken from the ``confidence`` attribute, and can be
            easily toggled off with this function's ``show_confidence`` option.
            But if you would like to alter the formatting of confidence values,
            or label the branches with something other than confidence, then use
            this option.
        label_colors : dict or callable
            A function or a dictionary specifying the color of the tip label.
            If the tip label can't be found in the dict or label_colors is
            None, the label will be shown in black.

    This function is adapted from Biopython for uses in ProDy.
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections as mpcollections

    # Arrays that store lines for the plot of clades
    horizontal_linecollections = []
    vertical_linecollections = []

    orientation = orientation.lower()
    if orientation == 'v':
        orientation = 'vertical'
    elif orientation == 'h':
        orientation = 'horizontal'

    # Options for displaying branch labels / confidence
    def conf2str(conf):
        if int(conf) == conf:
            return str(int(conf))
        return str(conf)

    if not branch_labels:
        if show_confidence:

            def format_branch_label(clade):
                try:
                    confidences = clade.confidences
                    # phyloXML supports multiple confidences
                except AttributeError:
                    pass
                else:
                    return "/".join(conf2str(cnf.value) for cnf in confidences)
                if clade.confidence is not None:
                    return conf2str(clade.confidence)
                return None

        else:

            def format_branch_label(clade):
                return None

    elif isinstance(branch_labels, dict):

        def format_branch_label(clade):
            return branch_labels.get(clade)

    else:
        if not callable(branch_labels):
            raise TypeError(
                "branch_labels must be either a dict or a callable (function)"
            )
        format_branch_label = branch_labels

    # options for displaying label colors.
    if label_colors:
        if callable(label_colors):

            def get_label_color(label):
                return label_colors(label)

        else:
            # label_colors is presumed to be a dict
            def get_label_color(label):
                return label_colors.get(label, "black")

    else:

        def get_label_color(label):
            # if label_colors is not specified, use black
            return "black"

    # Layout

    def get_x_positions(tree):
        """Create a mapping of each clade to its horizontal position.

        Dict of {clade: x-coord}
        """
        depths = tree.depths()
        # If there are no branch lengths, assume unit branch lengths
        if not max(depths.values()):
            depths = tree.depths(unit_branch_lengths=True)

        if inverted:
            max_depth = max(depths.values())
            for clade in depths:
                depths[clade] = max_depth - depths[clade]
        return depths

    def get_y_positions(tree):
        """Create a mapping of each clade to its vertical position.

        Dict of {clade: y-coord}.
        Coordinates are negative, and integers for tips.
        """
        maxheight = tree.count_terminals()
        # Rows are defined by the tips
        heights = {
            tip: maxheight - i for i, tip in enumerate(reversed(tree.get_terminals()))
        }

        # Internal nodes: place at midpoint of children
        def calc_row(clade):
            for subclade in clade:
                if subclade not in heights:
                    calc_row(subclade)
            # Closure over heights
            heights[clade] = (
                heights[clade.clades[0]] + heights[clade.clades[-1]]
            ) / 2.0

        if tree.root.clades:
            calc_row(tree.root)
        return heights

    x_posns = get_x_positions(tree)
    y_posns = get_y_positions(tree)
    # The function draw_clade closes over the axes object
    axes = plt.gca()

    def draw_clade_lines(
        here, start, stop,
        use_linecollection=False,
        orientation="horizontal",
        color="black",
        lw=.1,
    ):
        """Create a line with or without a line collection object.

        Graphical formatting of the lines representing clades in the plot can be
        customized by altering this function.
        """
        if not use_linecollection and orientation == "horizontal":
            axes.hlines(here, start, stop, color=color, lw=lw)
        elif use_linecollection and orientation == "horizontal":
            horizontal_linecollections.append(
                mpcollections.LineCollection(
                    [[(start, here), (stop, here)]], color=color, lw=lw
                )
            )
        elif not use_linecollection and orientation == "vertical":
            axes.vlines(here, start, stop, color=color)
        elif use_linecollection and orientation == "vertical":
            vertical_linecollections.append(
                mpcollections.LineCollection(
                    [[(here, start), (here, stop)]], color=color, lw=lw
                )
            )

    def draw_clade(clade, x_start, color, lw):
        """Recursively draw a tree, down from the given clade."""
        x_here = x_posns[clade]
        y_here = y_posns[clade]
        # phyloXML-only graphics annotations
        if hasattr(clade, "color") and clade.color is not None:
            color = clade.color.to_hex()
        if hasattr(clade, "width") and clade.width is not None:
            lw = clade.width * plt.rcParams["lines.linewidth"]
        # Draw a horizontal line from start to here
        orit = orientation
        draw_clade_lines(
            y_here, x_start, x_here,
            use_linecollection=True,
            orientation=orit,
            color=color,
            lw=lw,
        )
        # Add node/taxon labels
        label = label_func(clade)
        if label not in (None, clade.__class__.__name__):
            if orientation == 'horizontal':
                x = x_here
                y = y_here
                va = 'center'
                ha = 'right' if inverted else 'left'
            else:
                x = y_here
                y = x_here
                ha = 'center'
                va = 'top' if inverted else 'bottom'

            axes.text(x, y,
                " %s" % label,
                verticalalignment=va,
                horizontalalignment=ha,
                color=get_label_color(label))

        # Add label above the branch (optional)
        conf_label = format_branch_label(clade)
        if conf_label:
            if orientation == 'horizontal':
                x = 0.5 * (x_start + x_here)
                y = y_here
                va = 'center'
                ha = 'right' if inverted else 'left'
            else:
                x = y_here
                y = 0.5 * (x_start + x_here)
                ha = 'center'
                va = 'top' if inverted else 'bottom'

            axes.text(x, y,
                conf_label,
                fontsize="small",
                verticalalignment=va,
                horizontalalignment=ha)

        if clade.clades:
            orit = 'horizontal' if orientation=='vertical' else 'vertical'
            # Draw a vertical line connecting all children
            y_top = y_posns[clade.clades[0]]
            y_bot = y_posns[clade.clades[-1]]
            # Only apply widths to horizontal lines, like Archaeopteryx
            draw_clade_lines(
                x_here, y_bot, y_top,
                use_linecollection=True,
                orientation=orit,
                color=color,
                lw=lw,
            )
            # Draw descendents
            for child in clade:
                draw_clade(child, x_here, color, lw)

    start = max(x_posns.values()) if inverted else 0.
    draw_clade(tree.root, start, "k", plt.rcParams["lines.linewidth"])

    # If line collections were used to create clade lines, here they are added
    # to the pyplot plot.
    for i in horizontal_linecollections:
        axes.add_collection(i)
    for i in vertical_linecollections:
        axes.add_collection(i)

    # Aesthetics

    try:
        name = tree.name
    except AttributeError:
        pass
    else:
        if name:
            axes.set_title(name)
    
    if orientation == 'horizontal':
        axes.set_xlabel("branch length")
        axes.set_ylabel("taxa")
    else:
        axes.set_ylabel("branch length")
        axes.set_xlabel("taxa")
        
    # Add margins around the tree to prevent overlapping the axes
    xmax = max(x_posns.values())

    if orientation == 'horizontal':
        if inverted:
            axes.set_xlim(-0.2 * xmax, 1.05 * xmax)
        else:
            axes.set_xlim(-0.05 * xmax, 1.2 * xmax)
        axes.set_ylim(0.2, max(y_posns.values()) + 0.8)
    else:
        if inverted:
            axes.set_ylim(-0.2 * xmax, 1.05 * xmax)
        else:
            axes.set_ylim(-0.05 * xmax, 1.2 * xmax)
        axes.set_xlim(0.2, max(y_posns.values()) + 0.8)

    # Parse and process key word arguments as pyplot options
    for key, value in kwargs.items():
        try:
            # Check that the pyplot option input is iterable, as required
            list(value)
        except TypeError:
            raise ValueError(
                'Keyword argument "%s=%s" is not in the format '
                "pyplot_option_name=(tuple), pyplot_option_name=(tuple, dict),"
                " or pyplot_option_name=(dict) " % (key, value)
            ) from None
        if isinstance(value, dict):
            getattr(plt, str(key))(**dict(value))
        elif not (isinstance(value[0], tuple)):
            getattr(plt, str(key))(*value)
        elif isinstance(value[0], tuple):
            getattr(plt, str(key))(*value[0], **dict(value[1]))

    return x_posns, y_posns
