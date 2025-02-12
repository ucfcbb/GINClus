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
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D



def determine_cluster_no(kmin, kmax, kinc, unknown_motif_family_list, output_path):

    print("Calculating optimal value of K based on silhouette score and SMCR...")
    Sil_high = -1
    Cluster_no = kmin
    Feature_no = 128
    motif_family_no = len(unknown_motif_family_list)
    label_mapping = {}
    
    for fam_ind in range(motif_family_no):
        label_mapping[unknown_motif_family_list[fam_ind]] = fam_ind
    

    ################################################################################
    # Dataset
    ################################################################################
    #X1 = pd.read_csv(output_path + 'Motif_Features_IL.tsv', sep = '\t')
    X1 = pd.read_csv(os.path.join(output_path,'Motif_candidate_features.tsv'), sep = '\t')
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
    # pca = PCA(n_components = 2)
    # X1_principal = pca.fit_transform(X1_normalized)
    # X1_principal = pd.DataFrame(X1_principal)
    # X1_principal.columns = ['P1', 'P2']


    ################################################################################
    # K-means Clustering
    ################################################################################
    for k in range(kmin, kmax+1, kinc):
          
          kmeans = KMeans(n_clusters = k).fit(X1_normalized)
          labels = kmeans.labels_
          unique_value, unique_freq = np.unique(labels, return_counts=True)
          count = np.count_nonzero(unique_freq == 1)
          #print(unique_freq)
          #print("count: ", count)

          SMCR = (count / k) * 100
          sil_score = silhouette_score(X1_normalized, labels, metric = 'euclidean')
          print("For cluster number ", k, "silhouette score is: ", sil_score, " and SMCR is :", SMCR)

          if sil_score > Sil_high and SMCR < 5:
            Sil_high = sil_score
            Cluster_no = k

          if sil_score < Sil_high:
            print("Silhouette score has started to decrease, so stopping silhouette score calculation.")
            break
          
          if SMCR > 5:
            print("SMCR value has crossed 5%, so stopping silhouette score calculation.")
            break


    return Cluster_no



# def run_kmeans(unknown_motif_family_list, CLUSTER_NO, output_path):


#     Feature_no = 128
#     motif_family_no = len(unknown_motif_family_list)
#     label_mapping = {}
    
#     for fam_ind in range(motif_family_no):
#         label_mapping[unknown_motif_family_list[fam_ind]] = fam_ind
    


#     ################################################################################
#     # Dataset
#     ################################################################################
#     #X1 = pd.read_csv(output_path + 'Motif_Features_IL.tsv', sep = '\t')
#     X1 = pd.read_csv(os.path.join(output_path,'Motif_candidate_features.tsv'), sep = '\t')
#     X1['Label'] = X1['Label'].map(label_mapping).astype('int32')
#     family_labels = X1['Label'].tolist()
#     motif_ids = X1['Motif_id'].tolist()
#     X1 = X1.drop('Label', axis = 1)
#     X1 = X1.drop('Motif_id', axis = 1)



#     ################################################################################
#     # Data Pre-processing (Scaling, Normalize, PCA)
#     ################################################################################
#     # Scaling the data to bring all the attributes to a comparable level
#     scaler = StandardScaler()
#     X1_scaled = scaler.fit_transform(X1)

#     ### Normalizing the data so that the data approximately follows a Gaussian distribution
#     X1_normalized = normalize(X1_scaled)
     
#     ### Converting the numpy array into a pandas DataFrame
#     X1_normalized = pd.DataFrame(X1_normalized)
#     #print(X1_normalized.head())

#     ## Convert high dimension paraeters to 2 dimensiion
#     pca = PCA(n_components = 2)
#     X1_principal = pca.fit_transform(X1_normalized)
#     X1_principal = pd.DataFrame(X1_principal)
#     X1_principal.columns = ['P1', 'P2']



#     ################################################################################
#     # K-means Clustering
#     ################################################################################
#     #kmeans = KMeans(n_clusters=len(label_mapping), random_state=0)
#     kmeans = KMeans(random_state=0, n_clusters = CLUSTER_NO)

#     try:
#         kmeans_labels = kmeans.fit_predict(X1_normalized)
#     except ValueError as e:
#         print("Provided value of K (" + str(CLUSTER_NO) + ") is greater than the total number of samples (" + str(len(X1_scaled)) + "). Value of K has to be less than total number of samples. Set the parameter K to a value lower than " + str(len(X1_scaled)) + ".")
#         sys.exit()

#     #kmeans_labels = kmeans.fit_predict(X1_principal)
#     #print('Original Labels:', family_labels)
#     #print('Kmeans Labels:', kmeans_labels)
#     max_cluster = max(kmeans_labels)
#     #print(len(family_labels))
#     family_list = label_mapping.keys()
#     cluster_list = list(np.arange(max_cluster+1))



#     ################################################################################
#     # Output Clustering
#     ################################################################################
#     label_family = dict((v,k) for k,v in label_mapping.items())
#     output_file = os.path.join(output_path, "Cluster_output.csv")
#     Clus_out = open(output_file, "w")
#     #Clus_out = open(output_path + "Cluster_output.csv", "w")
    

#     ### Write header name
#     # Clus_out.write("%s\t" % ('Motif_location (' + input_index_type.upper() + ')'))
#     Clus_out.write("%s\t" % ('Motif_id'))
#     for feature in range(1, Feature_no+1):
#         cur_feature = 'Feature_' + str(feature)
#         Clus_out.write("%s\t" % (cur_feature))   
#     Clus_out.write("%s\t%s\n" % ('Family_label', 'Cluster_id'))
    
#     for i in range(len(motif_ids)):
        
#         Clus_out.write("%s\t" % (motif_ids[i]))
#         feature_list = X1.iloc[[i]].to_string(header=None, index=False)
#         # print(feature_list)
#         feature_list = feature_list.split()

#         # print(feature_list)
#         # sys.exit()

#         for f in feature_list:
#             Clus_out.write("%s\t" % (f))
#         Clus_out.write("%s\t%s\n" % (label_family[family_labels[i]], str(kmeans_labels[i])))
        
#     Clus_out.close()






    


