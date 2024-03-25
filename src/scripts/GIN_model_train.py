"""
Train the GIN model and save the best model weight
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
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GINConv, GlobalAvgPool, GCNConv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


tf.config.run_functions_eagerly(True)
################################################################################
# Config
################################################################################
learning_rate = 1e-3  # Learning rate
channels = 128  # Hidden units
layers = 3  # GIN layers
epochs = 1000  # Number of training epochs
es_patience = 300  # Patience for early stopping
##test_percen = 0.937
##val_percen = 0.873



##############################################################################
#Defining model
##############################################################################
def run_model(train_data_path, family_list, idxs_par, save_par, val_percen, test_percen):
    
    motif_family_no = len(family_list)
    family_dic = {}
    family_labels = {}

    data_list = []
    idxs = []


    for key in family_list:
        fam_count = 0
        for file in os.listdir(train_data_path):
            if file.startswith(key):
                fam_count += 1
        family_dic[key] = fam_count
    

    for fam_ind in range(len(family_list)):
        family_labels[family_list[fam_ind]] = fam_ind


    for key in family_dic:
        member_no = family_dic[key]
        for i in range(member_no):
            data_list.append(key + '_Graph_' + str(i))
            
        
    ################################################################################
    # Defining MyDataset class
    ################################################################################
    class MyDataset(Dataset):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)


        def read(self):
            
            output = []
              
            for key in family_dic:
                member_no = family_dic[key]

                for i in range(member_no):
                    if(True):        

                        x = []
                        a = []
                        e = []
                        y = np.zeros(motif_family_no)

                        ### Distance
                        e_distance = []  
                        index = 0
                        
                        ### Read data from file
                        fin = open(train_data_path + key + '_Graph_' + str(i) + '.g')

                        # Read Motif location
                        line = fin.readline()
                        PDB_loc = fin.readline()
                        FASTA_loc = fin.readline()
                        line = fin.readline()
                        

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
    # Load data
    ################################################################################
    dataset = MyDataset()
   
    # Parameters
    F = dataset.n_node_features  # Dimension of node features
    n_out = dataset.n_labels  # Dimension of the target
    
    # Split Dataset into Train, Evaluate and Test data
    if int(idxs_par) == 0:
        
        ### 6 Features Accuracy 85
        idxs = [260, 206, 247, 295, 185, 111, 232, 102, 77, 228, 301, 183, 288, 126, 84, 21, 230, 233, 251, 263, 279, 125, 175, 73, 9, 291, 304, 15, 282, 270, 116, 199, 142, 140, 173, 159, 168, 41, 40, 98, 123, 109,\
                14, 24, 293, 157, 207, 60, 277, 62, 261, 129, 65, 132, 190, 275, 120, 211, 312, 104, 250, 237, 259, 18, 278, 149, 145, 208, 218, 289, 216, 70, 64, 171, 188, 231, 61, 155, 66, 280, 227, 91, 201, 26,\
                76, 36, 35, 197, 86, 121, 13, 309, 45, 305, 186, 176, 300, 4, 58, 244, 203, 205, 264, 55, 214, 189, 283, 239, 241, 119, 158, 179, 292, 202, 169, 195, 167, 193, 88, 80, 204, 30, 112, 254, 44, 281, 87,\
                299, 81, 267, 110, 115, 287, 213, 222, 156, 93, 72, 2, 42, 310, 234, 39, 177, 32, 223, 6, 79, 184, 225, 180, 48, 29, 107, 137, 31, 212, 253, 131, 144, 174, 221, 20, 284, 83, 311, 143, 68, 146, 191,\
                242, 27, 165, 152, 154, 286, 153, 57, 71, 302, 114, 92, 49, 113, 1, 53, 194, 308, 210, 130, 122, 236, 127, 118, 37, 34, 141, 135, 94, 19, 67, 235, 78, 172, 217, 38, 63, 266, 257, 196, 240, 117, 75,\
                294, 162, 285, 170, 85, 46, 182, 229, 248, 133, 252, 10, 124, 166, 151, 273, 265, 69, 246, 219, 100, 90, 11, 271, 290, 56, 160, 187, 307, 148, 105, 297, 47, 256, 139, 274, 306, 138, 50, 269, 238, 298,\
                5, 97, 268, 258, 17, 245, 178, 96, 150, 163, 134, 0, 296, 224, 164, 59, 181, 220, 8, 276, 209, 89, 52, 255, 12, 243, 108, 54, 95, 215, 136, 272, 74, 262, 51, 161, 28, 200,\
                25, 3, 106, 128, 147, 23, 7, 82, 249, 16, 22, 43, 33, 226, 103, 303, 192, 101, 99, 198]

    elif int(idxs_par) == 1:
        idxs = np.random.permutation(len(dataset))

    elif int(idxs_par) == 2:

        train_list = []
        val_list = []
        test_list = []

        train_val_test_file = open('data/Train_Validate_Test_data_list.csv', 'r')

        line = train_val_test_file.readline()
        
        ### Read motif graph file names for training data 
        while(True):
            line = train_val_test_file.readline()
            if line[0] == "#":
                break
            motif_name = line.strip("\n")
            train_list.append(motif_name)

        ### Read motif graph file names for validation data
        while(True):
            line = train_val_test_file.readline()
            if line[0] == "#":
                break
            motif_name = line.strip("\n")
            val_list.append(motif_name)
            
        ### Read motif graph file names for test data
        while(True):
            line = train_val_test_file.readline()
            if line == "":
                break
            motif_name = line.strip("\n")
            test_list.append(motif_name)

        train_val_test_file.close()


        for motif in train_list:
            idxs.append(data_list.index(motif))
        for motif in val_list:
            idxs.append(data_list.index(motif))
        for motif in test_list:
            idxs.append(data_list.index(motif))            

    else:
        print("Please provide valid value for parameter idx")


        
    split_va, split_te = int(float(val_percen) * len(dataset)), int(float(test_percen) * len(dataset))
    idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])

    dataset_tr = dataset[idx_tr]
    dataset_va = dataset[idx_va]
    dataset_te = dataset[idx_te]
    

    ########################################################################################################
    tr_batch_size = len(dataset_tr) # Batch size
    va_batch_size = len(dataset_va)  # Batch size
    te_batch_size = len(dataset_te)  # Batch size
    ########################################################################################################

    loader_tr = DisjointLoader(dataset_tr, batch_size=tr_batch_size, epochs=epochs)
    loader_te = DisjointLoader(dataset_te, batch_size=te_batch_size, epochs=1)
    loader_va = DisjointLoader(dataset_va, batch_size=va_batch_size)
    loader_full = DisjointLoader(dataset, batch_size=len(dataset), epochs=1, shuffle = False)


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

    model = GIN0(channels, layers)
    optimizer = Adam(learning_rate)
    loss_fn = CategoricalCrossentropy()


    ################################################################################
    # Fit model
    ################################################################################
    @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, 1, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = tf.reduce_mean(categorical_accuracy(target, predictions))
        return loss, acc


    ################################################################################
    # Evaluate model
    ################################################################################
    def evaluate(loader):
        output = []
        step = 0
        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            pred = model(inputs, 1, training=False)
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
            output.append(outs)
            if step == loader.steps_per_epoch:
                output = np.array(output)
                return np.average(output[:, :-1], 0, weights=output[:, -1])


    ### Fit and Validate Model
    best_val_loss = np.inf
    best_weights = None
    patience = es_patience
    epoch = step = 0
    results = []
    for batch in loader_tr:
        step += 1
        loss, acc = train_step(*batch)
        results.append((loss, acc))
        if step == loader_tr.steps_per_epoch:
            step = 0
            epoch += 1
##            print("Ep. {} - Loss: {}. Acc: {}".format(epoch, *np.mean(results, 0)))

            # Compute validation loss and accuracy
            val_loss, val_acc = evaluate(loader_va)
##            print(
##                "Ep. {} - Loss: {:.6f} - Acc: {:.6f} - Val loss: {:.6f} - Val acc: {:.6f}".format(
##                    epoch, *np.mean(results, 0), val_loss, val_acc
##                )
##            )

            # Check if loss improved for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = es_patience
##                print("New best val_loss {:.6f}".format(val_loss))
                best_weights = model.get_weights()
            else:
                patience -= 1
                if patience == 0:
##                    print("Early stopping (best val_loss: {})".format(best_val_loss))
                    break

            results = []


    ################################################################################
    # Evaluate model
    ################################################################################

    ### Metric Evaluation ###
    y_test = []
    y_pred = []
    results = []


    model.set_weights(best_weights)  # Load best model
    for batch in loader_te:
        inputs, target = batch
        predictions = model(inputs, 1, training=False)
       

        ### Metric Evaluation ###
        y_test = target
        y_pred = predictions
        tf.nn.softmax(y_pred).numpy()
        y_pred  = y_pred.numpy()
        
        results.append(
            (
                loss_fn(target, predictions),
                tf.reduce_mean(categorical_accuracy(target, predictions)),
            )
        )

    #print("Done. Test loss: {}. Test acc: {}".format(*np.mean(results, 0)))
   


    ##############################################################################
    #Saving Model Weight
    ##############################################################################
    if int(save_par) == 1:
        path = 'model_weight/best_weights'
        model.save_weights(path)








