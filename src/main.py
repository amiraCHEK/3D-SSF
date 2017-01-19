#!/usr/bin/python
import os
import sys
sys.path.append("/home/mira/Bureau/3D-SSF/tools")
import numpy
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from nibabel import trackvis as tv
import pickle
from dipy.io.pickles import save_pickle
from dipy.io.pickles import load_pickle
from mid import midpt 
import dipy.segment.quickbundles  as oldqb
import dipy.segment.clustering as newqb
from dipy.tracking.metrics import downsample
from sklearn.metrics.pairwise import euclidean_distances
from dipy.tracking.metrics import midpoint
from metric import dist_translation
from dipy.viz import fvtk
from initialization_phase import initialization
from Summary_statistics_maintenance_clustering_phase import online

################################### Data Loading ##############################################################
print 'Data Loading .....'
#brain_name = "/home/mira/Bureau/3D-SSF/data-test/bundle1"
brain_name = "/home/mira/Bureau/3D-SSF/data-test/bundle2"
streams, hdr = tv.read(brain_name)
streamlines=[downsample(s[0],21) for s in streams]
############################################################ parameters  #################################################################



# Runtime parameters
sample        = len(streamlines)
NBRofIteration      = 1 
epsilon=0.1
lamda=6
sigma=80

####################################################################################################################################


class basic_agent:
    """This class defines the basic agent and the rules of behavoir"""
    def __init__(self, index):
        self.cluster = None
        self.visited = False
        self.index = index
        self.pos = midpt(streamlines[index])
        self.streamline= streamlines[index]
        self.vel    = numpy.zeros(3)
        self.size   = numpy.random.rand(1)

    def vector (self, feature_vector):
        self.feature_vector = feature_v

    def limitvelocity(self,maxvel):
        """Limiting the speed to avoid unphysical jerks in motion"""
        if numpy.linalg.norm(self.vel) > maxvel:
            self.vel = (self.vel/numpy.linalg.norm(self.vel))*maxvel



class potential_agent:
    """This class defines the basic agent and the rules of behavoir"""
    def __init__(self, index):
        self.cluster = None
        self.visited = False
        self.index  = index
        
        self.pos = midpt(swarms_potential[index])
        self.streamline= swarms_potential[index]
        self.vel    = numpy.zeros(3)
        self.size   = numpy.random.rand(1)
    def vector (self, feature_vector):
        self.feature_vector = feature_v

    def limitvelocity(self,maxvel):
        """Limiting the speed to avoid unphysical jerks in motion"""
        if numpy.linalg.norm(self.vel) > maxvel:
            self.vel = (self.vel/numpy.linalg.norm(self.vel))*maxvel

class outlier_agent:
    """This class defines the basic agent and the rules of behavoir"""
    def __init__(self, index):
        self.cluster = None
        self.visited = False
        self.index  = index
        
        self.pos = midpt(swarms_outlier[index])
        self.streamline = swarms_outlier[index]
        self.vel    = numpy.zeros(3)
        self.size   = numpy.random.rand(1)
   
    def vector (self, feature_vector):
        self.feature_vector = feature_v

    def limitvelocity(self,maxvel):
        """Limiting the speed to avoid unphysical jerks in motion"""
        if numpy.linalg.norm(self.vel) > maxvel:
            self.vel = (self.vel/numpy.linalg.norm(self.vel))*maxvel
    




#################################### initialization phase ##########################################################
tab=[]
#Generate the flock of the 1000 first streamlines
flock_initial           = [basic_agent(count) for count in range(sample)]
        

############## Compute summary statisctics of the initialization phase ###########################      

final_indice_potential=[]
final_indice_outlier=[]
print 'Initialization phase ......'
for i in range(NBRofIteration):
    tab=initialization(i, flock_initial)

qb=oldqb.QuickBundles(tab,15,21)
medoids,moyenne=qb.exemplars(tab)
swarms_potential_cluster=[]
swarms_outlier_cluster=[]
swarms_outlier_cluster0=[]
swarms_potential=[]
swarms_outlier=[]
p_representative=[]
o_representative=[]
indices=[]
cluster=[]
final_potential=[]
distance=numpy.zeros(len(medoids))
qbbb1=oldqb.QuickBundles(medoids,500.,21)
medoid_medoid,ind=qbbb1.exemplars(medoids)
medoid_medoid=numpy.asarray(medoid_medoid)
for k, medoid in enumerate (medoids):
    distance[k] =dist_translation(medoid, medoid_medoid)
thershold= (numpy.sum(distance)/len(medoids))*1.5

for k in range(len(moyenne)):
    indices=qb.clustering[k]['indices']
    for i in indices:
        cluster.append (tab[i])
        
    
    if len(cluster)>((sample)/len(moyenne)) or distance[k]<thershold:
        swarms_potential.append(medoids[k])
        final_potential.append(medoids[k])
        final_indice_potential.append(moyenne[k])
    else:
        swarms_outlier.append(medoids[k])
        final_indice_outlier.append(moyenne[k])


###################### Online Phase ###################### 

nqb = newqb.QuickBundles(threshold=50)
medoids=[]
new_sample=sample+100
print 'Summary statistics maintenance and clustering phase ......'
while new_sample < (len(streamlines)):
    swarms_potential_cluster=[]
    swarms_outlier_cluster=[]
    
    flock           = [basic_agent(count) for count in range(sample,new_sample)]
    
    potential_flock = [potential_agent(count) for count in range(len (swarms_potential))]
    outlier_flock = [outlier_agent(count) for count in range(len (swarms_outlier))]
    
    for i in range(NBRofIteration):
        
        swarms_potential_cluster, swarms_outlier_cluster, swarms_potential, swarms_outlier = online(i,flock) 


    #print '\n========== Results of online phase =============

    inds_potential=[]
    medoids=[]
    for swarm in swarms_potential_cluster:    
        qbb2=oldqb.QuickBundles(swarm,500.,21)
        medoid,ind_potential=qbb2.exemplars(swarm)
        ind_potential=numpy.add(ind_potential, sample)
        for k in medoid :
            medoids.append(k)
        for i in ind_potential :
            inds_potential.append(i)
        
    
    qbb3=oldqb.QuickBundles(medoids,500.,21)
    medoid_medoid,ind=qbb3.exemplars(medoids)
    medoid_medoid=numpy.asarray(medoid_medoid)
    distance=numpy.zeros(len(medoids))
    for k, medoid in enumerate (medoids):
        distance[k] =dist_translation(medoid, medoid_medoid)
    thershold= ((numpy.sum(distance))/len(medoids))*1.5
    #print len(medoids)
    for swarm in swarms_potential_cluster:
        for k, medoid in enumerate (medoids):
            if len(swarm)>(100/len(swarms_potential_cluster)) or distance[k]<thershold:
                swarms_potential.append(medoids[k])
                final_potential.append(medoids[k])
                final_indice_potential.append(inds_potential[k])
            else:
                swarms_outlier.append(medoids[k])
                final_indice_outlier.append(inds_potential[k])
    inds_outlier=[]
    medoids=[]
    for swarm in swarms_outlier_cluster:
        qbb4=oldqb.QuickBundles(swarm,500.,21)
        medoid,ind_outlier=qbb4.exemplars(swarm)
        ind_outlier=numpy.add(ind_outlier, sample)

        for k in medoid :
            medoids.append(k)
        
        for i in ind_outlier:
            inds_outlier.append(i)
    
    qbb5=oldqb.QuickBundles(medoids,500.,21)
    medoid_medoid,ind_ind=qbb5.exemplars(medoids)
    medoid_medoid=numpy.asarray(medoid_medoid)
    distance=numpy.zeros(len(medoids))
    for k, medoid in enumerate (medoids):
        distance[k] =dist_translation(medoid, medoid_medoid)
    thershold= (numpy.sum(distance)/len(medoids))*1.5
    for swarm in swarms_outlier_cluster:
        for k, medoid in enumerate(medoids):
            if len(swarm)>(100/len(swarms_outlier_cluster)) or distance[k]<thershold:
                swarms_potential.append(medoids[k])
                final_potential.append(medoids[k])
                final_indice_potential.append(inds_outlier[k])
            else:
                swarms_outlier.append(medoids[k])
                final_indice_outlier.append(inds_outlier[k])

    sample=new_sample
    new_sample=sample+100       

               

############################ Save Final results in trk format ######################################################

final_potential_cluster=[]
final_potential_cluster_indices= nqb.cluster(final_potential)
for i  in range (len(final_potential_cluster_indices)):
    final_potential_cluster.append(final_potential_cluster_indices[i][:])


print 'Save final results in pkl format ......'
save_pickle('final_potential_bundles_pkl.pkl', final_potential_cluster)



print 'Save final results in trk format ......'
colormap_full = np.ones((len(final_potential), 3))
def save_streamlines_with_color(streamlines, colors, out_file, hdr):
    trk = []

    for i, streamline in enumerate(streamlines):
        # Color (RBG [0-255]) for each point of the streamline
        scalars = np.array([colors[i]] * len(streamline)) * 255
        properties = None
        trk.append((streamline, scalars, properties))

    tv.write(out_file, trk, hdr)

save_streamlines_with_color(final_potential, colormap_full , 'final_potential_bundles_trk.trk' , hdr)




