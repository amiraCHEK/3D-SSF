
import numpy

Dimensions          = 3
def normalize(vector):
    """Normalizes a vector"""

    vector = vector/numpy.linalg.norm(vector)
    #print vector
    return vector

def centerofpos(flock):
    
    """Calculates the 'center of mass' for the flock of basic agents"""
   
    com = numpy.zeros(Dimensions)   
    for dim in range(Dimensions):
        for basic_agent in flock:
            
            com[dim] = com[dim] + basic_agent.pos[dim]
        com[dim] = com[dim]/len(flock)
    return com


def centerofvel(flock):
    """Calculates the mean velocity vector for the flock of boids"""
    cov = numpy.zeros(Dimensions)   
    for dim in range(Dimensions):
        for basic_agent in flock:

            cov[dim] = cov[dim] + basic_agent.vel[dim]
        cov[dim] = cov[dim]/len(flock)
    return cov