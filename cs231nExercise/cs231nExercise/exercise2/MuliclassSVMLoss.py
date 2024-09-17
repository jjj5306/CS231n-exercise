import numpy as np


def L_i_vectorized(x, y, W):
    (" A faster half-vectorized implementation. \n"
     "half-vectorized refers to the fact that for a single example the implementation contains no for loops,\n"
     "but there is still one loop over the examples. (outside this function)")

    delta = 1.0 # margin
    scores = W.dot(x) # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta.
    # We want to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
