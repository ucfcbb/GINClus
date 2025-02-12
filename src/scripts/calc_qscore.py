import os
import sys
import copy
import networkx as nx
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt



def read_max_min_val(input_file_location):

    MAX_SC_Score = 0
    MAX_SC_RMSD = 0
    MAX_SC_AL = 0
    MAX_TM_Score = 0
    MAX_TM_RMSD = 0
    MAX_TM_AL = 0
    
    infile = open(input_file_location, "r")
    header = infile.readline()

    while(True):
        line = infile.readline()
        if line == "":
            break

        line = line.split(",")
        subclus, pdb1, pdb2, SC_score, SC_RMSD, SC_AL, TM_score, TM_RMSD, TM_AL = line[0], line[1], line[2], float(line[3]), float(line[4]), float(line[5]), float(line[8]), float(line[9]), float(line[10])

        if SC_score > MAX_SC_Score: MAX_SC_Score = SC_score
        if SC_RMSD > MAX_SC_RMSD: MAX_SC_RMSD = SC_RMSD
        if SC_AL > MAX_SC_AL: MAX_SC_AL = SC_AL
        if TM_score > MAX_TM_Score: MAX_TM_Score = TM_score
        if TM_RMSD > MAX_TM_RMSD: MAX_TM_RMSD = TM_RMSD
        if TM_AL > MAX_TM_AL: MAX_TM_AL = TM_AL

    return MAX_SC_Score, MAX_SC_RMSD, MAX_SC_AL, MAX_TM_Score, MAX_TM_RMSD, MAX_TM_AL



def calc_qscore_for_each_subcluster(SUBCLUSTER_NO):

    SUBCLUS_Dic = {}
    SUBCLUS_motif_dic = {}
    
    input_file_location = 'output/temp/Subcluster_output_qscore_temp.csv'
    output_file_location = 'output/Subcluster_output_qscore.csv'

    infile = open(input_file_location, "r")
    header = infile.readline()

    MAX_SC_Score, MAX_SC_RMSD, MAX_SC_AL, MAX_TM_Score, MAX_TM_RMSD, MAX_TM_AL = read_max_min_val(input_file_location)

    while(True):
        line = infile.readline()
        if line == "":
            break

        line = line.split(",")
        subclus, pdb1, pdb2, SC_score, SC_RMSD, SC_AL, TM_score, TM_RMSD, TM_AL = line[0], line[1], line[2], float(line[3]), float(line[4]), float(line[5]), float(line[8]), float(line[9]), float(line[10])

        ### Normalizing values
        SC_score_norm = (SC_score - 0)/(MAX_SC_Score - 0)
        SC_RMSD_norm = (SC_RMSD - 0)/(MAX_SC_RMSD - 0)
        SC_AL_norm = (SC_AL - 0)/(MAX_SC_AL - 0)
        TM_score_norm = TM_score
        TM_RMSD_norm = (TM_RMSD - 0)/(MAX_TM_RMSD - 0)
        TM_AL_norm = (TM_AL - 0)/(MAX_TM_AL - 0)
        
        subclus_no = int(subclus.split("_")[1])

        if subclus_no not in SUBCLUS_Dic:
            SUBCLUS_Dic[subclus_no] = [0, 0, 0, 0, 0, 0, 0]
            SUBCLUS_motif_dic[subclus_no] = []
       
        SUBCLUS_Dic[subclus_no][0] = SUBCLUS_Dic[subclus_no][0] + SC_score_norm
        SUBCLUS_Dic[subclus_no][1] = SUBCLUS_Dic[subclus_no][1] + SC_RMSD_norm
        SUBCLUS_Dic[subclus_no][2] = SUBCLUS_Dic[subclus_no][2] + SC_AL_norm
        SUBCLUS_Dic[subclus_no][3] = SUBCLUS_Dic[subclus_no][3] + TM_score_norm
        SUBCLUS_Dic[subclus_no][4] = SUBCLUS_Dic[subclus_no][4] + TM_RMSD_norm
        SUBCLUS_Dic[subclus_no][5] = SUBCLUS_Dic[subclus_no][5] + TM_AL_norm
        SUBCLUS_Dic[subclus_no][6] = SUBCLUS_Dic[subclus_no][6] + 1

        if pdb1 not in SUBCLUS_motif_dic[subclus_no]:
            SUBCLUS_motif_dic[subclus_no].append(pdb1)

    Cluster_qscore = []

    MAX_Cluster_size = 0
    for key in SUBCLUS_motif_dic:
        cur_len = len(SUBCLUS_motif_dic[key])
        if cur_len > MAX_Cluster_size: MAX_Cluster_size = cur_len

    for key in SUBCLUS_Dic:

        if SUBCLUS_Dic[key][6] != 0:
            AVG_SC_SCORE = SUBCLUS_Dic[key][0] / SUBCLUS_Dic[key][6]
            AVG_SC_RMSD = SUBCLUS_Dic[key][1] / SUBCLUS_Dic[key][6]
            AVG_SC_AL = SUBCLUS_Dic[key][2] / SUBCLUS_Dic[key][6]
            AVG_TM_SCORE = SUBCLUS_Dic[key][3] / SUBCLUS_Dic[key][6]
            AVG_TM_RMSD = SUBCLUS_Dic[key][4] / SUBCLUS_Dic[key][6]
            AVG_TM_AL = SUBCLUS_Dic[key][5] / SUBCLUS_Dic[key][6]
            NO_motifs = len(SUBCLUS_motif_dic[key])
            NO_motifs_norm = NO_motifs / MAX_Cluster_size 

            ### Calculate Q-score:
            Scanx_Q_score = AVG_SC_SCORE + AVG_SC_AL + NO_motifs_norm - AVG_SC_RMSD
            TM_Q_score = AVG_TM_SCORE + AVG_TM_AL + NO_motifs_norm - AVG_TM_RMSD
            Q_score = (Scanx_Q_score + TM_Q_score)/2
            Cluster_qscore.append((key, Q_score))

    sorted_cluster_qscore = sorted(Cluster_qscore, key=lambda x: x[1], reverse=True)
    
    infile.close()

    ### Write output in a CSV file
    outfile = open(output_file_location, "w")

    outfile.write("Subcluster_id,Q-score\n")

    for clus in sorted_cluster_qscore:
        
        subclus = clus[0]
        qscore = clus[1]
        outfile.write("%s,%s\n" % (subclus, qscore))
        
    outfile.close()
