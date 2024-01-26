import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
from config import *

sys.path.append('src/scripts/')

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import metrics

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def run_subclustering(CLUSTER_NO, unknown_motif_family_list, output_path):

    ################################################################################
    # Dataset
    ################################################################################
    X = pd.read_csv(output_path + 'Cluster_output.csv', sep = '\t')
    
    motif_family_no = len(unknown_motif_family_list)
    label_mapping = {}
    
    for fam_ind in range(motif_family_no):
        label_mapping[unknown_motif_family_list[fam_ind]] = fam_ind
   
    label_family = dict((v,k) for k,v in label_mapping.items())

    X['Family_label'] = X['Family_label'].map(label_mapping).astype('int32')


    subclus_no = -1
    subclus_dic = {}


    ### Formatting Clustering + subclustering output
    f_subclus = open(output_path + "Subcluster_output.csv","w")
    f_subclus.write("%s\t%s\t%s\t%s\n" % ('Motif_location (' + input_index_type.upper() + ')', 'Cluster_id', 'Subcluster_id', 'Family_label'))


    ### Generate Subcluster for each cluster
    for clus in range(CLUSTER_NO):    
        X1 = X[X['Cluster_id'] == clus]
        X1 = X1.drop('Cluster_id', axis = 1)
        family_labels = X1['Family_label'].tolist()
        X1 = X1.drop('Family_label', axis = 1)
        motif_ids = X1['Motif_id'].tolist()
        X1 = X1.drop('Motif_id', axis = 1)
        
        ################################################################################
        # Data Pre-processing (Scaling, Normalize, PCA)
        ################################################################################
        # Scaling the data to bring all the attributes to a comparable level
        scaler = StandardScaler()
        X1_scaled = scaler.fit_transform(X1)
        
        ### Normalizing the data so that the data approximately follows a Gaussian distribution
        X1_normalized = normalize(X1_scaled)
         
        ### Converting the numpy array into a pandas DataFrame
        X1_normalized = pd.DataFrame(X1_normalized)
        
        #print('Original Labels:', family_labels)
        family_list = label_mapping.keys()

        if len(motif_ids) == 1:
            clus_subclus = str(clus) + '_0'
            subclus_no += 1
            subclus_dic[clus_subclus] = subclus_no
            f_subclus.write("%s\t%s\t%s\t%s\n" % (motif_ids[0], clus, subclus_dic[clus_subclus], label_family[family_labels[0]]))
            continue

        ################################################################################
        # Agglomerative Hierarchical Clustering for Normalized
        ################################################################################
        agglo = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=5.4)
        agglo_labels = agglo.fit_predict(X1_normalized)
        agglo_max_cluster = max(agglo_labels)
        agglo_cluster_list = list(np.arange(agglo_max_cluster+1))

        ################################################################################
        # Output Normalized Clustering
        ################################################################################
        for i in range(len(motif_ids)):
            clus_subclus = str(clus) + '_' + str(agglo_labels[i])
            if clus_subclus not in subclus_dic:
                subclus_no += 1
                subclus_dic[clus_subclus] = subclus_no
            
            f_subclus.write("%s\t%s\t%s\t%s\n" % (motif_ids[i], clus, subclus_dic[clus_subclus], label_family[family_labels[i]]))
        
    f_subclus.close()















