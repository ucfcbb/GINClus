"""
Generates features using trained GIN model
"""
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import shutil
import random
import sys

sys.path.append('../../')
from config import *

sys.path.append('src/scripts/')


# import warnings
# warnings.simplefilter("ignore")



from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GINConv, GlobalAvgPool, GCNConv



################################################################################
# Config
################################################################################
learning_rate = 1e-3  # Learning rate
channels = 128  # Hidden units
layers = 3  # GIN layers
epochs = 1000  # Number of training epochs
es_patience = 300  # Patience for early stopping




def generate_feature(INPUT_PATH, unknown_motif_family_list, output_path):


    motif_family_no = len(unknown_motif_family_list)
    motif_list = []
    family_labels = {}


    for fam_ind in range(motif_family_no):
        family_labels[unknown_motif_family_list[fam_ind]] = fam_ind

 
    ################################################################################
    # Load data
    ################################################################################
    class MyDataset(Dataset):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def read(self):
            
            output = []
            
            for filename in os.listdir(INPUT_PATH):
                   
                x = []
                a = []
                e = []
                y = np.zeros(motif_family_no)

                cur_filename = filename.split('.')[0]
                
                
                ### Distance
                e_distance = []  
                index = 0
                
                # Read data from file
                fin = open(INPUT_PATH + '/' + filename)

                # Read Motif location
                line = fin.readline()
                PDB_loc = fin.readline()
                FASTA_loc = fin.readline()
                line = fin.readline()


                cur_motif = FASTA_loc.split("\t")[0]
                if input_index_type == 'pdb':
                    cur_motif = PDB_loc.split("\t")[0]
                    
                motif_list.append(cur_motif)
                
                # Read Adjacency matrix
                line = fin.readline()
                
                while(True):
                    adj = fin.readline()
                    if adj[0] == '#':
                        break
                    adj = adj.strip('\n').split('\t')
                    adj = adj[0:len(adj)-1]
                   
                    ### Distance
                    adj = list(map(float, adj))
                    for i in range(len(adj)):
                       e_distance.append(adj[i])
                       adj[i] = 1
                    adj = np.asarray(adj)
                        
                    a.append(adj)
                a = np.asarray(a)

                # Read node features
                while(True):
                    node_feat = fin.readline()
                    if node_feat[0] == '#':
                        break
                    node_feat = node_feat.strip('\n').split('\t')
                    node_feat = node_feat[0:len(node_feat)-1]
                    node_feat = np.asarray(list(map(float, node_feat)))
                    x.append(node_feat)
                x = np.asarray(x)

                # Read edge features
                while(True):
                    edge_feat = fin.readline()
                    
                    if edge_feat[0] == '#':
                        break
                    edge_feat = edge_feat.strip('\n').replace('[', '').replace(']', '').split('\t')
                    edge_feat = edge_feat[0:len(edge_feat)-1]
                    
                    ### Distance
                    edge_feat = list(map(float, edge_feat))
                    edge_feat.insert(0, e_distance[index])
                    index += 1
                    edge_feat = np.asarray(edge_feat)
                    
                    e.append(edge_feat)
                e = np.asarray(e)

                graph_label = fin.readline()
                graph_label = graph_label.strip('\n')
                graph_label = family_labels[graph_label]
                y[graph_label] = 1

        
                output.append(
                    Graph(x=x, a=a, y=y, e=e)
                )
               
                fin.close()

            return output


    ################################################################################
    # Create Dataset
    ################################################################################
    dataset = MyDataset()
    loader_full = DisjointLoader(dataset, batch_size=len(dataset), epochs=1, shuffle=False)


    # Parameters
    F = dataset.n_node_features  # Dimension of node features
    n_out = dataset.n_labels  # Dimension of the target


    ################################################################################
    # Build model
    ################################################################################
    class GIN0(Model):
        def __init__(self, channels, n_layers):
            super().__init__()
            self.conv1 = GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
         
            self.convs = []
            for _ in range(1, n_layers):
                self.convs.append(
                    GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
                )

            self.pool = GlobalAvgPool()
            #self.dense1 = Dense(channels, activation="relu")
            self.dense1 = Dense(channels, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
            self.dropout = Dropout(0.2)
            self.dense2 = Dense(n_out, activation="softmax")

        def call(self, inputs, run):
            x, a, e, i = inputs
            x = self.conv1([x, a, e])
            for conv in self.convs:
                x = conv([x, a, e])
            x = self.pool([x, i])
            x = self.dense1(x)
            x = self.dropout(x)
            if run == 1:
                return self.dense2(x)
            else:
                return x


    # Build model
    model = GIN0(channels, layers)
    optimizer = Adam(learning_rate)
    loss_fn = CategoricalCrossentropy()


    ################################################################################
    # Feature Generation
    ################################################################################
    ### Feature File Header
    # header = ['Motif_location (' + input_index_type.upper() + ')']
    header = ['Motif_id']
    for i in range(1, channels+1):
        feature_head = 'Feature_' + str(i)
        header.append(feature_head)
    header.append('Label')
        

    ################################################################################
    # Loading Model Weight
    ################################################################################
    path = 'model_weight/best_weights'
    model.load_weights(path).expect_partial()
    #model.set_weights(best_weights)  # Load best model


    for batch in loader_full:
        inputs, target = batch
        predictions = model(inputs, 2, training=False)
       
        ### Write Features In TSV File
        features = predictions.numpy()    
        fcsv = open(output_path + "Motif_Features_IL.tsv", "w")
        header = str(header)
        header = header.strip('[').strip(']').replace("'", "").replace(',', '\t').replace(" ", "")
        fcsv.write("%s\n" % (header))

        label_family = dict((v,k) for k,v in family_labels.items())

        for i in range(len(features)):
            for j in range(len(target[i])):
                if target[i][j]: target_family = label_family[j]
            current_features = str(list(features[i]))
            current_features = current_features.strip('[').strip(']').replace(',', '\t')
            fcsv.write("%s\t%s\t%s\n" % (motif_list[i], current_features, str(target_family)))

        fcsv.close()

