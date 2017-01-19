from __future__ import division

import numpy as np
import utils

from dipy.tracking.metrics import downsample
from dipy.tracking.distances import bundles_distances_mdf



def dist_translation(s1, s2):
    """ Compute translation distance between two streamlines.
        The distance used is the Euclidean distance between
        mass center of the two streamlines.

    Parameters
    ----------
    s1 : (N, 3) array_like
         streamline
    s2 : (N, 3) array_like
         streamline

    Returns
    -------
    dist : float
        Translation distance between `s1` and `s2`

    """

    return utils.norm(s1.mean(0) - s2.mean(0))


def dist_rotation(s1, s2):
    """ Compute rotation distance between two streamlines.
        The distance used is the cosine distance between
        first principal component of the two streamlines.

    Parameters
    ----------
    s1 : (N, 3) array_like
         streamline
    s2 : (N, 3) array_like
         streamline

    Returns
    -------
    dist : float
        Rotation distance between `s1` and `s2`

    """

    components_s1, projection_s1, importance_s1 = utils.princomp(s1, 3)
    components_s2, projection_s2, importance_s2 = utils.princomp(s2, 3)

    #Compute cosinus distance between the first principal component of s1 and s2
    return 1 - np.abs(np.dot(components_s1[:, 0], components_s2[:, 0]))


def dist_scaling(s1, s2):
    """ Compute scaling distance between two streamlines.
        The distance used is simply the difference between
        the arc length of the two streamlines.

    Parameters
    ----------
    s1 : (N, 3) array_like
         streamline
    s2 : (N, 3) array_like
         streamline

    Returns
    -------
    dist : float
        Scaling distance between `s1` and `s2`

    """

    length_s1 = np.sum(utils.norm(s1[1:, :] - s1[:-1, :], axis=1))
    length_s2 = np.sum(utils.norm(s2[1:, :] - s2[:-1, :], axis=1))

    return np.abs(length_s1 - length_s2)


def scaling(s1):
    """ Compute the arc length of the streamline.

    Parameters
    ----------
    s1 : (N, 3) array_like
         streamline
    

    Returns
    -------
    length : float
        arc length of the streamline

    """

    length_s1 = np.sum(utils.norm(s1[1:, :] - s1[:-1, :], axis=1))
    

    return length_s1 


def MDF(s1, s2):
    """ Compute shape distance between two streamlines.
        The distance used is the ``MDF`` distance by [1]_
        between the two streamlines.

    Parameters
    ----------
    s1 : (N, 3) array_like
         streamline
    s2 : (N, 3) array_like
         streamline

    Returns
    -------
    dist : float
        Shape distance between `s1` and `s2`

    References
    ----------
    .. [1] E. Garyfallidis, M. Brett,  M. M. Correia, G. B. Williams and I. Nimmo-Smith,
           *QuickBundles, a method for tractography simplification*,
           Frontiers in Neuroscience, 2012, #175 vol. 6

    """

    resampled_s1 = downsample(s1, 12)
    resampled_s2 = downsample(s2, 12)

    return bundles_distances_mdf(resampled_s1, resampled_s2)

def magn(xyz,n=1):
    ''' magnitude of vector

    '''
    mag=np.sum(xyz**2,axis=1)**0.5
    imag=np.where(mag==0)
    mag[imag]=np.finfo(float).eps

    if n>1:
        return np.tile(mag,(n,1)).T
    return mag.reshape(len(mag),1)


def frenet_serret(xyz):
    

    ''' Frenet-Serret Space Curve Invariants

    Calculates the 3 vector and 2 scalar invariants of a space curve
    defined by vectors r = (x,y,z).  If z is omitted (i.e. the array xyz has
    shape (N,2), then the curve is
    only 2D (planar), but the equations are still valid.

    Similar to
    http://www.mathworks.com/matlabcentral/fileexchange/11169

    In the following equations the prime ($'$) indicates differentiation
    with respect to the parameter $s$ of a parametrised curve $\mathbf{r}(s)$.

    - $\mathbf{T}=\mathbf{r'}/|\mathbf{r'}|\qquad$ (Tangent vector)}

    - $\mathbf{N}=\mathbf{T'}/|\mathbf{T'}|\qquad$ (Normal vector)

    - $\mathbf{B}=\mathbf{T}\times\mathbf{N}\qquad$ (Binormal vector)

    - $\kappa=|\mathbf{T'}|\qquad$ (Curvature)

    - $\mathrm{\tau}=-\mathbf{B'}\cdot\mathbf{N}$ (Torsion)

    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track

    Returns
    -------
    T : array shape (N,3)
        array representing the tangent of the curve xyz
    N : array shape (N,3)
        array representing the normal of the curve xyz
    B : array shape (N,3)
        array representing the binormal of the curve xyz
    k : array shape (N,1)
        array representing the curvature of the curve xyz
    t : array shape (N,1)
        array representing the torsion of the curve xyz

   '''



    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')

    dxyz=np.gradient(xyz)[0]
    ddxyz=np.gradient(dxyz)[0]
    #Tangent
    T=np.divide(dxyz,magn(dxyz,3))
    #Derivative of Tangent
    dT=np.gradient(T)[0]
    #Normal
    N = np.divide(dT,magn(dT,3))
    #Binormal
    B = np.cross(T,N)
    #Curvature
    k = magn(np.cross(dxyz,ddxyz),1)/(magn(dxyz,1)**3)
    #Torsion
    #(In matlab was t=dot(-B,N,2))
    t = np.sum(-B*N,axis=1)
    #return T,N,B,k,t,dxyz,ddxyz,dT
    return T,N,B,k,t

def mean_curvature(xyz):
    ''' Calculates the mean curvature of a curve

    Parameters
    ------------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a curve

    Returns
    -----------
    m : float
        Mean curvature.
 
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')

    dxyz=np.gradient(xyz)[0]
    ddxyz=np.gradient(dxyz)[0]

    #Curvature
    k = magn(np.cross(dxyz,ddxyz),1)/(magn(dxyz,1)**3)

    return np.mean(k)

def mean_orientation(xyz):
    '''
    Calculates the mean orientation of a curve

    Parameters
    ------------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a curve

    Returns
    -------
    m : float
        Mean orientation.
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')

    dxyz=np.gradient(xyz)[0]

    return np.mean(dxyz,axis=0)





