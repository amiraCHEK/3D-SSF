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


def online (NBRofIteration, flock):


    """This is the calculation timeloop"""

   
    Timestep =str(NBRofIteration)
    
    
    
    ##################### potentiel agent meet another potentiel agent
    
    
    for potential_agent in potential_flock:
        
        #print potential_agent
        
        # Agent RULE 1. Cohesion - Steer to move towoards the center of mass
        cohesion = numpy.zeros(Dimensions)
        cohesion = (centerofpos(potential_flock) - potential_agent.pos)*WeightCenterOfMass
        # Agent RULE 3. Alignment - Steer towards the average heading of local flockmates
        alignment = numpy.zeros(Dimensions)
        alignment = (centerofvel(potential_flock)-potential_agent.vel)*WeightAlignment
      
        # compute the similarity
        euclidean_distance=numpy.zeros(Dimensions)
        Vsim=numpy.zeros(Dimensions)
        separation = numpy.zeros(Dimensions)
        

        for potential_agent2 in potential_flock:
            if potential_agent != potential_agent2:
                euclidean_distance = math.sqrt((potential_agent.pos[0]-potential_agent2.pos[0])**2+(potential_agent.pos[1]-potential_agent2.pos[1])**2+(potential_agent.pos[2]-potential_agent2.pos[2])**2)
                dif  = potential_agent2.pos - potential_agent.pos
                dist= numpy.linalg.norm(dif) 
               
                if euclidean_distance <= threshold_similarity:
                     
                    # Agent  RULE 2. Separation - steer to avoid crowding local flockmates
                    if dist< MinSeparation :
                        separation = separation - normalize(dif)/dist
                    separation = separation*WeightSeparation 
                    Vsim= euclidean_distance * dist
                    potential_agent.vel = separation + cohesion+ alignment+Vsim
                    potential_agent.limitvelocity(MaxVelocity)
                    #potential_agent.pos = potential_agent.pos + potential_agent.vel
                    potential_agent.streamline = numpy.sum([potential_agent.streamline , potential_agent.vel], axis=0)
                        
                elif euclidean_distance > threshold_similarity :
                    Vsim= 1/(euclidean_distance * dist)
                    potential_agent.vel = separation + cohesion+ alignment+Vsim
                    potential_agent.limitvelocity(MaxVelocity)
                    #potential_agent.pos = potential_agent.pos + potential_agent.vel
                    potential_agent.streamline = numpy.sum([potential_agent.streamline , potential_agent.vel], axis=0)
    for potential_agent in potential_flock:
        if not numpy.isnan(potential_agent.streamline).all():
            swarms_potential.append(potential_agent.streamline)
            
          
    swarms_potential_cluster_indices= nqb.cluster(swarms_potential)
    
    for i  in range (len(swarms_potential_cluster_indices)):
        swarms_potential_cluster.append(swarms_potential_cluster_indices[i][:])
    
    ################### outlier agent meet another outlier agent
    
    for  outlier_agent in outlier_flock:
          
        # Agent RULE 1. Cohesion - Steer to move towoards the center of mass
        cohesion = numpy.zeros(Dimensions)
        cohesion = (centerofpos(outlier_flock) - outlier_agent.pos)*WeightCenterOfMass
        # Agent RULE 3. Alignment - Steer towards the average  of local flockmates
        alignment = numpy.zeros(Dimensions)
        alignment = (centerofvel(outlier_flock) - outlier_agent.vel)*WeightAlignment
        # compute the similarity
        euclidean_distance=numpy.zeros(Dimensions)
        Vsim=numpy.zeros(Dimensions)
        separation = numpy.zeros(Dimensions)
        for outlier_agent2 in outlier_flock:
            if outlier_agent != outlier_agent2:
                euclidean_distance = math.sqrt((outlier_agent.pos[0]-outlier_agent2.pos[0])**2+(outlier_agent.pos[1]-outlier_agent2.pos[1])**2+(outlier_agent.pos[2]-outlier_agent2.pos[2])**2)
                dif = outlier_agent2.pos - outlier_agent.pos
                dist= numpy.linalg.norm(dif)
                if euclidean_distance <= threshold_similarity :
                    
                    # Agent  RULE 2. Separation - steer to avoid crowding local flockmates
                    if dist< MinSeparation :
                        separation = separation - normalize(dif)/dist
                    separation = separation*WeightSeparation 
                    Vsim= euclidean_distance * dist
                    outlier_agent.vel = separation + cohesion+ alignment+Vsim
                    outlier_agent.limitvelocity(MaxVelocity)
                    outlier_agent.streamline = numpy.sum([outlier_agent.streamline , outlier_agent.vel], axis=0)   
                        
                elif euclidean_distance > threshold_similarity :
                    Vsim= 1/(euclidean_distance * dist)
                    outlier_agent.vel = separation + cohesion+ alignment+Vsim
                    outlier_agent.limitvelocity(MaxVelocity)
                    outlier_agent.streamline = numpy.sum([outlier_agent.streamline , outlier_agent.vel], axis=0)
    for outlier_agent in outlier_flock:
        if not numpy.isnan(outlier_agent.streamline).all():
            swarms_outlier.append(outlier_agent.streamline)
      
    swarms_outlier_cluster_indices= nqb.cluster(swarms_outlier)
   
    for i  in range (len(swarms_outlier_cluster_indices)):
        
        swarms_outlier_cluster.append(swarms_outlier_cluster_indices[i][:])
    
    
        ################### basic agent meet another potentiel agent or outlier agent faire le lien with if
    
    for basic_agent in flock:
        
        pot_added = False
        out_added = False
    
        for swarm_potential in swarms_potential_cluster:
        
            for potential_agent in swarm_potential:
                if not numpy.isnan(potential_agent).all(): 
                    mid_potential=midpt(potential_agent)
                    euclidean_distance = math.sqrt((basic_agent.pos[0]-mid_potential[0])**2+(basic_agent.pos[1]-mid_potential[1])**2+(basic_agent.pos[2]-mid_potential[2])**2)
                    if euclidean_distance <= threshold_similarity:
                        pot_added = True
                        swarm_potential.append(basic_agent.streamline)
                        numpy.delete(flock, basic_agent.streamline)
                        
                        break
        if not pot_added and (len(swarms_outlier)!=0):
        
            for swarm_outlier in swarms_outlier_cluster:
                for outlier_agent in swarm_outlier:
                    if not numpy.isnan(outlier_agent).all():
                        mid_outlier=midpt(outlier_agent)
                        euclidean_distance = math.sqrt((basic_agent.pos[0]-mid_outlier[0])**2+(basic_agent.pos[1]-mid_outlier[1])**2+(basic_agent.pos[2]-mid_outlier[2])**2)
                        if euclidean_distance <= threshold_similarity:
                            out_added = True
                        
                            swarm_outlier.append(basic_agent.streamline)
                            
                            numpy.delete(flock, basic_agent.streamline)
                            break
    
    
        if ( not out_added) and ( not pot_added):
         
   
            ################### basic agent meet another basic agent
            # Agent RULE 1. Cohesion - Steer to move towoards the center of mass
            cohesion = numpy.zeros(Dimensions)
            cohesion = (centerofpos(flock) - basic_agent.pos)*WeightCenterOfMass
            # Agent RULE 3. Alignment - Steer towards the average heading of local flockmates
            alignment = numpy.zeros(Dimensions)
            alignment = (centerofvel(flock) - basic_agent.vel)*WeightAlignment
            euclidean_distance=numpy.zeros(Dimensions)
            Vsim=numpy.zeros(Dimensions)
            separation = numpy.zeros(Dimensions)
    
            for basic_agent2 in flock:
                if basic_agent != basic_agent2:
                    euclidean_distance = math.sqrt((basic_agent.pos[0]-basic_agent2.pos[0])**2+(basic_agent.pos[1]-basic_agent2.pos[1])**2+(basic_agent.pos[2]-basic_agent2.pos[2])**2)
                    dif  = basic_agent2.pos - basic_agent.pos
                    dist= numpy.linalg.norm(dif)
                    
                    if euclidean_distance < threshold_similarity:
                        
                        # Agent  RULE 2. Separation - steer to avoid crowding local flockmates
                        if dist < MinSeparation:
                            separation = separation - normalize(dif)/dist
                        separation = separation*WeightSeparation 
                        Vsim= euclidean_distance * dist
                        basic_agent.vel = separation + cohesion+ alignment+Vsim
                        basic_agent.limitvelocity(MaxVelocity)
                        
                        basic_agent.streamline = numpy.sum([basic_agent.streamline , basic_agent.vel], axis=0)
                
        
        
            swarm_basic_agent=[basic_agent.streamline for basic_agent in flock]
            swarms_outlier_cluster0_indices= nqb.cluster(swarm_basic_agent)
            for i  in range (len(swarms_outlier_cluster0_indices)):
                swarms_outlier_cluster.append(swarms_outlier_cluster0_indices[i][:]) 
                        


        # return at the end of iteration
    return swarms_potential_cluster, swarms_outlier_cluster, swarms_potential, swarms_outlier

