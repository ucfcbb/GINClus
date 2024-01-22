import argparse
import sys
sys.path.append('src/scripts/')
import os
import glob
import logging
import time


from GIN_model_train import *
from GIN_feature_gen import *
from Clustering_IL import *
from Subclustering_IL import *




def main():

    process_start_time = time.time()
    parser = argparse.ArgumentParser(description='Cluster RNA motifs based on structural similarity')

    parser.add_argument('-it', nargs='?', default='graphs_dir/Train_Graph_Data/', const='graphs_dir/Train_Graph_Data/', help="Path to the graph data generated for training motifs. Default: 'graphs_dir/Train_Graph_Data/'.")
    parser.add_argument('-il', nargs='?', default='graphs_dir/IL_Graph_Data/', const='graphs_dir/IL_Graph_Data/', help="Path to the graph data generated for unknown internal loop (IL) motifs. Default: 'graphs_dir/IL_Graph_Data/'.")
    parser.add_argument('-o', nargs='?', default='output/', const='output/', help="Path to the output files. Default: 'output/'.")
    parser.add_argument('-t', nargs='?', default='True', const='True', help="Trains the model if t = True, else uses the previously trained model weight if t = False. Default: 'True'.")
    parser.add_argument('-idx', nargs='?', default='0', const='0', help="Training data permutation. To use random data permutation, use '1'. To use data permutation used in the paper, use '0'. Default: '0'.")
    parser.add_argument('-w', nargs='?', default='1', const='1', help="Paraameter to save the best model weight. Use '1' to save the new best model weight. Otherwise, use '0'. Default: '1'.")
    parser.add_argument('-val', nargs='?', default='0.873', const='0.873', help="Sets the number '1-val' as the percentage of data that should be considered as validation data while training the model. Default: '0.873'.")
    parser.add_argument('-test', nargs='?', default='0.937', const='0.937', help="Sets the number '1-test' as the percentage of data that should be considered as test data while training the model. Default: '1'.")
    parser.add_argument('-f', nargs='?', default='True', const='True', help="Generates features for unknown motifs if f = True, else uses the previously generated features if f = False. Default: 'True'.")
    parser.add_argument('-c', nargs='?', default='True', const='True', help="Generates cluster output if t = True, else uses the previously generated clustering output if t = False. Default: 'True'.")
    parser.add_argument('-k', nargs='?', default='400', const='400', help="Define the number of clusters (value of K) to be generated. Default: 400.")
    parser.add_argument('-tf', nargs='?', default='0', const='0', help="If tf = 1, takes RNA motif family information (family name and nname prefix) for train data from a file named 'Train_family_info.csv' inside 'data' folder. Otherwise uses the existing family information for training. Default: '0'.")
    parser.add_argument('-uf', nargs='?', default='0', const='0', help="This parameter is useful if there are some known family motifs in the unknown motif family folder. If uf = 1, takes RNA motif family information (family name and name prefix) for unknown motif data from a file named 'Unknown_motif_family_info.csv' inside 'data' folder. If all the motifs have unknown family, then use uf = 2. Otherwise, uses the existing family information for unknown motif data. Default: '0'.")

    

    try:
        args = parser.parse_args()
    except Exception as e:
        parser.print_help()
        sys.exit()

    train_model = args.t
    train_family_info = args.tf
    train_data_path = args.it
    idxs_par = args.idx
    save_par = args.w
    val_percen = args.val
    test_percen = args.test
   
    gen_feature = args.f
    il_data_path = args.il
    unknown_motif_family_info = args.uf

    cluster = args.c
    CLUSTER_NO = int(args.k)

    output_path = args.o
    

    ### Reading motif family name and prefixes for train motif data
    if train_family_info == '1':
        family_list = []
        fam_info_file = open('data/Train_family_info.csv', 'r')

        line = fam_info_file.readline()
        
        while (True):

            line = fam_info_file.readline()
            if line == "":
                break

            fam_prefix = line.strip('\n').split(',')[1]
            family_list.append(fam_prefix)

        print(family_list)
            
        fam_info_file.close()
    else:
        family_list = ['CL', 'EL', 'HT', 'KT', 'SR', 'TS']


    ### Reading motif family name and prefixes for unknown motif data
    if unknown_motif_family_info == '1':
        unknown_motif_family_list = []
        unknown_fam_info_file = open('data/Unknown_motif_family_info.csv', 'r')

        line = unknown_fam_info_file.readline()
        
        while (True):

            line = unknown_fam_info_file.readline()
            if line == "":
                break

            fam_prefix = line.strip('\n').split(',')[1]
            unknown_motif_family_list.append(fam_prefix)

        print(unknown_motif_family_list)
            
        unknown_fam_info_file.close()
        
    elif unknown_motif_family_info == '2':
        unknown_motif_family_list = ['IL']
        
    else:
        unknown_motif_family_list = ['CL', 'EL', 'HT', 'L1C', 'KT', 'rKT', 'RS', 'SR', 'TL', 'TR', 'TS', 'IL']
        
        
    
    ### Train the model
    if train_model == 'True':
        print('Training GIN model...')
        run_model(train_data_path, family_list, idxs_par, save_par)
    else:
        print('Using model weight from previously trained model...')
    

    ### Generate features for unknown motifs
    if gen_feature == 'True':
        print('Generating features for unknown motifs...')
        generate_feature(il_data_path, unknown_motif_family_list, output_path)
    else:
        print('Using previously generated feature sets for unknown motifs...')

        
    ### Cluster unknown motifs
    if cluster == 'True':
        print('Clustering unknown motifs...')
        run_kmeans(unknown_motif_family_list, CLUSTER_NO, output_path)
    else:
        print('Using previously clustered unknown motifs output...')


    ### Subcluster unknown motifs
    run_subclustering(CLUSTER_NO, unknown_motif_family_list, output_path)
    


if __name__ == '__main__':
    
    main()
