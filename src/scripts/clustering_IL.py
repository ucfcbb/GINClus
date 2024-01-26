import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
from config import *

sys.path.append('src/scripts/')

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D





def run_kmeans(unknown_motif_family_list, CLUSTER_NO, output_path):


    Feature_no = 128
    motif_family_no = len(unknown_motif_family_list)
    label_mapping = {}
    
    for fam_ind in range(motif_family_no):
        label_mapping[unknown_motif_family_list[fam_ind]] = fam_ind
    


    ################################################################################
    # Dataset
    ################################################################################
    #X1 = pd.read_csv('Motif_Features_IL_95_2.tsv', sep = '\t')
    X1 = pd.read_csv(output_path + 'Motif_Features_IL.tsv', sep = '\t')
    X1['Label'] = X1['Label'].map(label_mapping).astype('int32')
    family_labels = X1['Label'].tolist()
    motif_ids = X1['Motif_id'].tolist()
    X1 = X1.drop('Label', axis = 1)
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
    #print(X1_normalized.head())

    ## Convert high dimension paraeters to 2 dimensiion
    pca = PCA(n_components = 2)
    X1_principal = pca.fit_transform(X1_normalized)
    X1_principal = pd.DataFrame(X1_principal)
    X1_principal.columns = ['P1', 'P2']



    ################################################################################
    # K-means Clustering
    ################################################################################
    #kmeans = KMeans(n_clusters=len(label_mapping), random_state=0)
    kmeans = KMeans(random_state=0, n_clusters = CLUSTER_NO)
    kmeans_labels = kmeans.fit_predict(X1_normalized)
    #kmeans_labels = kmeans.fit_predict(X1_principal)
    #print('Original Labels:', family_labels)
    #print('Kmeans Labels:', kmeans_labels)
    max_cluster = max(kmeans_labels)
    #print(len(family_labels))
    family_list = label_mapping.keys()
    cluster_list = list(np.arange(max_cluster+1))



    ################################################################################
    # Output Clustering
    ################################################################################
    label_family = dict((v,k) for k,v in label_mapping.items())

    Clus_out = open(output_path + "Cluster_output.csv", "w")
    

    ### Write header name
    # Clus_out.write("%s\t" % ('Motif_location (' + input_index_type.upper() + ')'))
    Clus_out.write("%s\t" % ('Motif_id'))
    for feature in range(1, Feature_no+1):
        cur_feature = 'Feature_' + str(feature)
        Clus_out.write("%s\t" % (cur_feature))   
    Clus_out.write("%s\t%s\n" % ('Family_label', 'Cluster_id'))
    
    for i in range(len(motif_ids)):
        
        Clus_out.write("%s\t" % (motif_ids[i]))
        feature_list = X1.iloc[[i]].to_string(header=None, index=False)
        # print(feature_list)
        feature_list = feature_list.split()

        # print(feature_list)
        # sys.exit()

        for f in feature_list:
            Clus_out.write("%s\t" % (f))
        Clus_out.write("%s\t%s\n" % (label_family[family_labels[i]], str(kmeans_labels[i])))
        
    Clus_out.close()






    


