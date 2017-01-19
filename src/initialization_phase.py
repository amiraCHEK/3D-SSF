from flock_function import centerofpos
from flock_function import normalize
from flock_function import centerofvel
import numpy
import math
Dimensions          = 3
WeightCenterOfMass  = 0.03
WeightSeparation    = 1
WeightAlignment     = 0.125
WeightRandom        = 0.0
WeightCenter        = 0.02 #0.03
MaxVelocityP        = 1
MaxVelocity         = 1
threshold_similarity= 100
MinSeparation= 500

def initialization(NBRofIteration, flock_initial):

    """This is the calculation timeloop"""

    
    Timestep =str(NBRofIteration)
    
    

    
    for basic_agent in flock_initial:
        
        # Agent RULE 1. Cohesion - Steer to move towoards the center of mass
        cohesion = numpy.zeros(Dimensions)
        cohesion = (centerofpos(flock_initial) - basic_agent.pos)*WeightCenterOfMass
 
        # Agent  RULE 2. Separation - steer to avoid crowding local flockmates
        separation = numpy.zeros(Dimensions)
        for basic_agent2 in flock_initial:
            difference  = basic_agent2.pos - basic_agent.pos
            if basic_agent != basic_agent2 and difference.all()!= 0:
                
                distance    = numpy.linalg.norm(difference)
                if distance < MinSeparation:
                    separation = separation - normalize(difference)/distance
                separation = separation*WeightSeparation 
 
        # Agent RULE 3. Alignment - Steer towards the average heading of local flockmates
        alignment = numpy.zeros(Dimensions)
        alignment = (centerofvel(flock_initial) - basic_agent.vel)*WeightAlignment

        # compute the similarity
        euclidean_distance=numpy.zeros(Dimensions)
        Vsim=numpy.zeros(Dimensions)
        for basic_agent2 in flock_initial:
            euclidean_distance=math.sqrt((basic_agent.pos[0]-basic_agent2.pos[0])**2+(basic_agent.pos[1]-basic_agent2.pos[1])**2+(basic_agent.pos[2]-basic_agent2.pos[2])**2)
            
            dif  = basic_agent2.pos - basic_agent.pos
            if basic_agent != basic_agent2 and dif.all()!= 0:
                
                dist= numpy.linalg.norm(dif)
                if euclidean_distance < threshold_similarity:
                    Vsim= euclidean_distance * dist
                elif euclidean_distance > threshold_similarity:
                    Vsim= 1/(euclidean_distance * dist)

        # Move the agent
        basic_agent.vel = separation + cohesion+ alignment+Vsim
        basic_agent.limitvelocity(MaxVelocity)
        basic_agent.streamline = numpy.sum([basic_agent.streamline , basic_agent.vel], axis=0)
        
    tab=[basic_agent.streamline for basic_agent in flock_initial] 
    return tab