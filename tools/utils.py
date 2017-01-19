from __future__ import division

import numpy as np


def norm(A, order=2, axis=None):
    """ Compute norm of a vector/matrix.
        The norm used is determined by the order. One can specified an axis
        along which the norm will be computed.


    Parameters
    ----------
    A : (M,) or (M, N) array_like
        Input array
    order : {non-zero int, inf, -inf}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's `inf` object.
    axis : {None, non-zero int}, optional
        Axis along which the norm will be computed for matrices.

    Returns
    -------
    norm : float
        Norm of the matrix or vector.

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ================================
    order  norm for matrices along axis
    =====  ================================
    inf    max(abs(x), axis)
    -inf   min(abs(x), axis)
    0      sum(x != 0, axis)
    other  sum(abs(x)**ord, axis)**(1./ord)
    =====  ================================

    The Frobenius norm can be obtained by setting order=2 and axis=None (default).

    """
    if order == np.inf:
        return np.max(np.abs(A), axis=axis)

    if order == -np.inf:
        return np.min(np.abs(A), axis=axis)

    if order == 0:
        return np.sum(A != 0, axis=axis)

    return np.sum(np.abs(A) ** order, axis=axis) ** (1. / order)


def princomp(A, numpc=0):
    """ Find `numpc` principal components of a given matrix `A`.

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    numpc : int, optional
        Number of principal components to keep. 0 means keep all components.

    Returns
    -------
    coeff : eigenvectors, (M, `numpc`) array_like
        `numpc` eigenvectors of the matrix sorted by their associated
        eigenvalue in ascending order.
    score : (M, N) array_like
        Projection of the data in the new space.
    latent : eigenvalues, (`numpc`, ) array_like
        `numpc` eigenvalues of the matrix in ascending order.

    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A, 0)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.cov(M))
    p = np.size(coeff, axis=1)
    idx = np.argsort(latent)  # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:, idx]
    latent = latent[idx]  # sorting eigenvalues
    if numpc < p or numpc >= 0:
        coeff = coeff[:, range(numpc)]  # cutting some PCs
    score = np.dot(coeff.T, M)  # projection of the data in the new space
    return coeff, score, latent
