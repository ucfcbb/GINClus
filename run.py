import argparse
import sys
import os
import glob
import logging
import time
import math
import pickle

# logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import os
import shutil
import random
from spektral.data import Dataset, DisjointLoader, Graph

from config import *
sys.path.append(lib_dir)
sys.path.append(scripts_dir)
from my_log import *
from utils import *
from prepare_loops import *
from partial_pdb_generator import *
from cif import *
from image_utils import *

from GIN_model_train import *
from GIN_feature_gen import *
from clustering_IL import *
from subclustering_IL import *

import collections



def main():

    process_start_time = time.time()
    parser = argparse.ArgumentParser(description='Cluster RNA motifs based on structural similarity')

    # Graph generating input params
    parser.add_argument('-i1', nargs='?', default='Train_motif_location_IL_input_PDB.csv', const='Train_motif_location_IL_input_PDB.csv', help="Input file containing training motif locations. Default:'Train_motif_location_IL_input_PDB.csv'.")
    parser.add_argument('-i2', nargs='?', default='Unknown_motif_location_IL_input_PDB.csv', const='Unknown_motif_location_IL_input_PDB.csv', help="Input file containing motif candidate locations. Default:'Unknown_motif_location_IL_input_PDB.csv'.")
    parser.add_argument('-o', nargs='?', default='output/', const='output/', help="Path to the output files. Default: 'output/'.")
    parser.add_argument('-e', nargs='?', default='0', const='5', help="Number of extended residues beyond loop boundary to generate the loop.cif file. Default: 0.")
    parser.add_argument('-d', nargs='?', default='web', const='web', help="Use 'tool' to generate annotation from DSSR tool, else use 'web' to generate annotation from DSSR website. Default: 'web'.")
    parser.add_argument('-idt', nargs='?', default='pdb', const='pdb', help="Use 'fasta' if input motif index type is FASTA, else use 'pdb' if input motif index type is PDB. Default: 'pdb'.")


    # GIN model run and clustering input params
    parser.add_argument('-t', nargs='?', default=True, const=False, help="Trains the model if t = True, else uses the previously trained model weight. To set the parameter to False just use '-t'. Default: True.")
    parser.add_argument('-idx', nargs='?', default='0', const='0', help="Divides data into train, validation and test. To divide randomly, set to '0'. To divide according to the paper for internal loops, set to '1'. To divide according to the paper for hairpin loops, set to '2'. To define manually using the file 'Train_Validate_Test_data_list.csv' in data folder, set to '3'. Default: 0.")
    parser.add_argument('-w', nargs='?', default='1', const='1', help="Use '1' to save the new model weight, otherwise, use '0'. Default: '1'.")
    parser.add_argument('-val', nargs='?', default='0.064', const='0.064', help="Set the percentage of validation data. Default: '0.064'.")
    parser.add_argument('-test', nargs='?', default='0.063', const='0.063', help="Set the percentage of test data. Default: '0.063'.")
    parser.add_argument('-f', nargs='?', default=True, const=False, help="Generates features for unknown motifs if True, else uses the previously generated features. To set the parameter to False just use '-f'. Default: True.")
    parser.add_argument('-c', nargs='?', default=True, const=False, help="Generates cluster output if True, else uses the previously generated clustering output. To set the parameter to False just use '-c'. Default: True.")
    parser.add_argument('-k', nargs='?', default='400', const='400', help="Define the number of clusters (value of K) to be generated. Default: 400.")

    # pymol image generation params
    parser.add_argument('-p', nargs='?', default=False, const=True, help="If True, generates PyMOL images for output clusteres. Default: False.")



    try:
        args = parser.parse_args()
    except Exception as e:
        parser.print_help()
        sys.exit()

    user_input_fname1 = args.i1
    user_input_fname2 = args.i2
    use_loop_name_in_graph_fname = False
    loop_cif_extension = int(args.e)
    output_path = args.o
    dssr_type = args.d
    input_index_type = args.idt
    train_model = args.t
    idxs_par = args.idx
    save_par = args.w
    val_par = float(args.val)
    test_par = float(args.test)
    gen_feature = args.f
    cluster = args.c
    CLUSTER_NO = int(args.k)
    generate_images = args.p


    ### calculate val_percen and test_percen values
    test_percen = 1 - test_par
    val_percen = 1 - val_par - test_par

    ### create output directory
    root_dir = os.getcwd()
    output_path = os.path.join(root_dir, output_path)
    print(output_path)
    
    partial_pdbx_dir = os.path.join(data_dir, 'pdbx_extracted_ext' + str(loop_cif_extension))

    delete_directory(train_data_path)
    delete_directory(candidate_data_path)
    create_required_directories(partial_pdbx_dir, output_path)

    known_motif_family_list = prepare_data_and_generate_graphs(user_input_fname1, train_data_path, partial_pdbx_dir, loop_cif_extension, use_loop_name_in_graph_fname, dssr_type, input_index_type)
    logger.info('Graph generation for training complete.')
    unknown_motif_family_list = prepare_data_and_generate_graphs(user_input_fname2, candidate_data_path, partial_pdbx_dir, loop_cif_extension, use_loop_name_in_graph_fname, dssr_type, input_index_type)
    logger.info('Graph generation for input motifs complete.')

         
    print('')
    ### Train the model
    if train_model:
        print('Training GIN model...')
        print('')
        run_model(train_data_path, known_motif_family_list, idxs_par, save_par, val_percen, test_percen)
    else:
        print('Using model weight from previously trained model...')
        print('')
    

    ### Generate features for unknown motifs
    if gen_feature:
        print('Generating features for unknown motifs...')
        print('')
        generate_feature(candidate_data_path, unknown_motif_family_list, output_path, input_index_type)
    else:
        print('Using previously generated feature sets for unknown motifs...')
        print('')

        
    ### Cluster unknown motifs
    if cluster:
        print('Clustering unknown motifs...')
        print('')
        run_kmeans(unknown_motif_family_list, CLUSTER_NO, output_path)
    else:
        print('Using previously clustered unknown motifs output...')
        print('')


    ### Subcluster unknown motifs
    print('Subclustering unknown motifs...')
    print('')
    run_subclustering(CLUSTER_NO, unknown_motif_family_list, output_path, input_index_type)


    if generate_images == True:
        input_fname = os.path.join(output_path, 'Subcluster_output.csv')
        generate_motif_images(input_fname, output_path, partial_pdbx_dir, input_index_type)


    logger.info('Subcluster output generated.')
    logger.info('Process complete.')
    logger.info('\nTotal time taken: ' + str(round((time.time() - process_start_time), 3)) + ' seconds.\n')
        
def prepare_data_and_generate_graphs(user_input_fname, output_dir, partial_pdbx_dir, loop_cif_extension, use_loop_name_in_graph_fname, dssr_type, input_index_type):

    input_fname = os.path.join(data_dir, user_input_fname)
    input_fname_base = os.path.basename(input_fname)

    print('')
    logger.info('Reading input from ' + input_fname[base_path_len:])
    print('')

    families = {}
    fp_input = open(input_fname)
    loop_list = csv_to_list(fp_input.readlines())
    fp_input.close()

    loop_count = 0
    for item in loop_list:
        if len(item) > 1:
            # families[item[0]] = map(lambda x: str(strToNode(x)), item[1:]) # item[1:]
            families[item[0]] = item[1:]

    prepare_data(families, dssr_type)
    if input_index_type == 'pdb':
        families = convert_a_cluster_from_PDB_to_FASTA(families)
        for family_id in families:
            families[family_id] = list(map(lambda x: str(strToNode(x)), families[family_id]))

        ##### To save FASTA index from PDB index #####
        # fp = open('clusters_ML_FASTA.csv', 'w')
        # for fam_id in families:
        #     fp.write(fam_id+',')
        #     fp.write(','.join(families[fam_id]))
        #     fp.write('\n')
        # fp.close()
        # sys.exit()
        ##### To save FASTA index from PDB index #####


    ##### To save PDB index from FASTA index #####
    # if input_index_type == 'fasta':
    #     families = convert_a_cluster_from_FASTA_to_PDB(families)
    #     for family_id in families:
    #         families[family_id] = list(map(lambda x: str(x), families[family_id]))

    #     fp = open('Unknown_motif_location_ML_input_PDB.csv', 'w')
    #     for fam_id in families:
    #         fp.write(fam_id+',')
    #         fp.write(','.join(families[fam_id]))
    #         fp.write('\n')
    #     fp.close()
    #     sys.exit()
    ##### To save PDB index from FASTA index #####


    
    loop_count = 0
    loop_node_list_str = []
    for family_id in families:
        loops = families[family_id]
        loop_count += len(loops)
        for loop in loops:
            loop = str(strToNode(loop))
            loop_node_list_str.append(loop)
    
    duplicates = [item for item, count in collections.Counter(loop_node_list_str).items() if count > 1]
    if len(duplicates) > 0:
        print('duplicates:')
        print(duplicates)

    loop_node_list_str = sorted(list(set(loop_node_list_str)))

    #logger.info(str(loop_count) + ' loops (' + str(len(loop_node_list_str)) + ' unique) found in ' + str(len(families)) + ' famil' + ('ies' if len(families) > 1 else 'y') + '.')
    #print('')

    # get pdb and fasta files 
    # and
    # generate loop.cif files
    # and annotation files
    
    
    # prepare_data(families)
    prepare_loop_files(loop_node_list_str)    #chkd
    prepare_partial_pdbs(partial_pdbx_dir, loop_node_list_str, loop_cif_extension)

    f = open(user_input_fname + ".pickle","wb")
    pickle.dump(families, f)
    f.close()

    motif_family_list = generate_graphs_for_loops(user_input_fname, families, partial_pdbx_dir, output_dir, use_loop_name_in_graph_fname)
    return motif_family_list


def get_simplified_index(index_dict, index):
    if index not in index_dict:
        index_dict[index] = len(index_dict)

    return index_dict[index], index_dict

def distance_3d(x1, y1, z1, x2, y2, z2):
    d = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) + math.pow(z2 - z1, 2)* 1.0)
    # print("Distance is ")
    # print(d)
    return d

def get_three_letter_string_for_bp(interaction, orientation):
    return orientation[0] + ''.join(interaction.split('/'))

def get_reversed_interaction(interaction, interaction_type):
    if interaction_type == 'bp':
        return interaction[0] + interaction[1:][::-1]
    elif interaction_type == 'stack':
        if interaction == 'upward':
            return 'downward'
        elif interaction == 'downward':
            return 'upward'
        elif interaction == 'inward':
            # return 'outward'
            return interaction
        elif interaction == 'outward':
            # return 'inward'
            return interaction
        else:
            logger.error('Invalid stack interaction')
            sys.exit()
    else:
        # logger.error('Invalid interaction type')
        sys.exit()

def load_loop_data(loop):
    
    loop_fn = os.path.join(loop_dir, loop.replace(':', '_') + '.smf')
    fp = open(loop_fn)
    lines = fp.readlines()
    fp.close()

    sequence = lines[1].strip()
    joined_sequence = ''.join(sequence.strip().split('...'))

    # bps = []
    # stks = []
    bps = {}
    bp_cnt = 0
    stks = {}
    stk_cnt = 0

    reading_bp = True
    for line in lines[3:]:
        line = line.strip()
        pieces = line.split(',')

        if line == '#info=stacking':
            reading_bp = False
            continue

        s, e = pieces[0].strip().split('-')
        s = int(s)
        e = int(e)
        ntd_pair = joined_sequence[s] + joined_sequence[e]
        rev_ntd_pair = joined_sequence[e] + joined_sequence[s]

        if reading_bp == True:
            if s not in bps:
                bps[s] = {}
            if e not in bps[s]:
                bps[s][e] = []
            interaction = get_three_letter_string_for_bp(pieces[1].strip(), pieces[2].strip())
            bps[s][e].append((s, e, ntd_pair, interaction))
            if e not in bps:
                bps[e] = {}
            if s not in bps[e]:
                bps[e][s] = []
            # rev_ntd_pair = ntd_pair[1] + ntd_pair[0]
            rev_interaction = get_reversed_interaction(interaction, 'bp')
            bps[e][s].append((e, s, rev_ntd_pair, rev_interaction))
            bp_cnt += 1
        else:
            if s not in stks:
                stks[s] = {}
            if e not in stks[s]:
                stks[s][e] = []
            interaction = pieces[1].strip()
            stks[s][e].append((s, e, ntd_pair, interaction))
            if e not in stks:
                stks[e] = {}
            if s not in stks[e]:
                stks[e][s] = []
            rev_interaction = get_reversed_interaction(interaction, 'stack')
            stks[e][s].append((e, s, rev_ntd_pair, rev_interaction))
            stk_cnt += 1

    return joined_sequence, bps, bp_cnt, stks, stk_cnt

def get_one_hot_encoded_nucl(nucl):
    encoded_data = []
    nucl_list = ['A', 'C', 'G', 'U']
    for item in nucl_list:
        if item == nucl:
            encoded_data.append(1.0)
        else:
            encoded_data.append(0.0)

    return encoded_data

def get_encoded_interaction(interaction):
    interaction_list = ['cWW', 'cWH', 'cHW', 'cWS', 'cSW', 'cHH', 'cHS', 'cSH', 'cSS', 'tWW', 'tWH', 'tHW', 'tWS', 'tSW', 'tHH', 'tHS', 'tSH', 'tSS', 'upward', 'downward', 'inward', 'outward']
    if interaction in interaction_list:
        return interaction_list.index(interaction)

def get_one_hot_encoded_interaction(interaction):
    # print('interaction')
    # print(interaction)
    interaction_list = ['cWW', 'cWH', 'cHW', 'cWS', 'cSW', 'cHH', 'cHS', 'cSH', 'cSS', 'tWW', 'tWH', 'tHW', 'tWS', 'tSW', 'tHH', 'tHS', 'tSH', 'tSS', 'upward', 'downward', 'inward', 'outward']
    
    encoded_data = []
    
    for i in range(len(interaction_list)):
        encoded_data.append(0)

    if interaction in interaction_list:
        encoded_data[interaction_list.index(interaction)] = 1

    # print('returning:')
    # print(encoded_data)
    # sys.exit()
    return encoded_data

# def get_reversed_interaction(interaction, interaction_type):
#     if interaction_type == 'bp':
#         return interaction[0] + interaction[1:][::-1]
#     elif interaction_type == 'stack':
#         if interaction == 'upward':
#             return 'downward'
#         elif interaction == 'downward':
#             return 'upward'
#         elif interaction == 'inward':
#             # return 'outward'
#             return interaction
#         elif interaction == 'outward':
#             # return 'inward'
#             return interaction
#         else:
#             logger.error('Invalid stack interaction')
#             sys.exit()
#     else:
#         # logger.error('Invalid interaction type')
#         sys.exit()

def load_pdb_res_map(chain):
    """load sequence index->pdb index"""
    """{ref_index: (chain_id, pdb_index)}"""
    ret = {}
    # map_dir = '../nrPDBs_Old' # the directory for the mapping data
    fp = open(os.path.join(pdb_fasta_mapping_dir, get_modified_chain_id_if_any_lowercase_letter(chain)+'.rmsx.nch'))
    for line in fp.readlines():
        decom = line.strip().split('\t')
        ##################### for PDB #####################
        # if decom[0][0] == "'":
        #     ret[int(decom[1])] = (decom[0][1], decom[0][3:].replace('.', ''))
        # else:
        #     ret[int(decom[1])] = (decom[0][0], decom[0][1:].replace('.', ''))
        ##################### for PDB #####################
        ##################### for PDBx ####################
        if decom[0][0] == "'":
            chain_id = decom[0][1:].strip().split("'")[0]
            i = len(chain_id)+2
        else:
            chain_id = re.split('-?(\d+)',decom[0])[0]
            i = len(chain_id)

        if decom[0][-1].isalpha():
            icode = decom[0][-1]
            j = len(decom[0])-2
        else:
            icode = ''
            j = len(decom[0])

        seqnum = decom[0][i:j]
        ret[int(decom[1])] = (chain_id, seqnum, icode)
        ##################### for PDBx ####################
    return ret

def pdb_pos_map(pdb_res_map, m):
    """position in pdb index alignment"""
    ret = []
    for i in m:
        if i in pdb_res_map:
            ret.append(pdb_res_map[i])
        # if 
        else:
            # ret.append('na')
            ret.append(('', '', ''))
            logger.warning('!!!!!!!!!!!!!!!!!!!!!ALERT: APPENDING EMPTY TUPLE (NA) !!!!!!!!!!!!!!!!!!!!')

    return ret

def centroid(coord_list):
    if len(coord_list) > 0:
        return list(map(lambda z: 1.*z/len(coord_list), reduce(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2]), coord_list)))
    return None

def get_atom_coordinate(pdb_fn, residue_list):

    backbone_atoms, sugar_atoms = get_backbone_and_sugar_atoms()
    pdb_id = os.path.basename(pdb_fn)[:4]

    parser = FastMMCIFParser()
    structure = parser.get_structure('struct', pdb_fn)

    backbone = {}
    sugar = {}

    for chain_id, index, icd in residue_list:
        # if chain_id == 'n' and index == 'a':
        if chain_id == '':
            continue
        chain = structure[0][chain_id]
        residues = chain.get_residues()

        my_residues = {}

        for r in residues:
            hetflag, resseq, icode = r.get_id()
            my_residues[(resseq, icode)] = r

        i = int(index)
        icode = icd if len(icd) > 0 else ' '

        if (i, icode) not in my_residues:
            # ret.append(0)
            backbone[(pdb_id, chain_id, index, icd)] = [0., 0., 0.]
            sugar[(pdb_id, chain_id, index, icd)] = [0., 0., 0.]
        else:
            atom_coord = []
            for atom in backbone_atoms:
                if atom in my_residues[(i, icode)]:
                    atom_coord.append(my_residues[(i, icode)][atom].get_vector())

            backbone[(pdb_id, chain_id, index, icd)] = centroid(atom_coord)

            atom_coord = []
            for atom in sugar_atoms:
                if atom in my_residues[(i, icode)]:
                    atom_coord.append(my_residues[(i, icode)][atom].get_vector())

            sugar[(pdb_id, chain_id, index, icd)] = centroid(atom_coord)

    return backbone, sugar, structure

def get_pdb_index_list(lp):
    pdb_chain, regions = lp.split(':')
    # segments = regions.strip().split('_')
    # index_list = []
    r = list(map(lambda x: x.split('-'), regions.split('_')))
    index_list = reduce(lambda y, z: y+z, list(map(lambda x: list(range(int(x[0]), int(x[1])+1)), r)))
    pdb_res_map = load_pdb_res_map(pdb_chain)
    pdb_index_list = pdb_pos_map(pdb_res_map, index_list)

    return pdb_index_list

def generate_graphs_for_loops(user_input_fname, families, partial_pdbx_dir, output_dir, use_loop_name_in_graph_fname):
    graph_list = []
    for encoded_fam_id, family_id in enumerate(families):
        # coord_dict = {}
        loops = families[family_id]
        for loop in loops:
            loop = str(strToNode(loop))
            index_dict = {}
            pdb_pm = get_pdb_index_list(loop)
            coord_backbone, coord_sugar, pdb_structure = get_atom_coordinate(os.path.join(partial_pdbx_dir, loop.replace(':', '_') + '.cif'), pdb_pm)

            for ind in sorted(coord_backbone):
                simplified_index, index_dict = get_simplified_index(index_dict, ind)

            # for ind in sorted(coord_sugar):
            #     simplified_index, index_dict = get_simplified_index(index_dict, ind)

            # print(sorted(index_dict.items(), key=lambda item: item[1]))

            adj_matrix = []
            for i in range(len(index_dict)):
                temp_list = []
                for j in range(len(index_dict)):
                    temp_list.append(0.0)
                adj_matrix.append(temp_list)


            for ind1 in sorted(coord_backbone):
                for ind2 in sorted(coord_backbone):
                    dist = distance_3d(coord_backbone[ind1][0], coord_backbone[ind1][1], coord_backbone[ind1][2], coord_backbone[ind2][0], coord_backbone[ind2][1], coord_backbone[ind2][2])
                    # print(index_dict[ind1], index_dict[ind2], dist)
                    adj_matrix[index_dict[ind1]][index_dict[ind2]] = dist
                    # adj_matrix[index_dict[ind1]][index_dict[ind2]] = 1

            # adj_matrix = sp.csr_matrix(adj_matrix)
            adj_matrix = np.array(adj_matrix)
            # print(adj_matrix)
            # print(sp.csr_matrix(adj_matrix))
            # sys.exit()
                
            loop_fn = os.path.join(loop_dir, loop.replace(':', '_') + '.smf')
            if not os.path.exists(loop_fn):
                missing_loop_fn = os.path.join(loop_dir, 'missing_res', get_loop_type(loop), loop.replace(':', '_') + '.smf')
                if os.path.exists(missing_loop_fn):
                    logger.warning('Loop contains significant amount of missing residues. Skipping loop: ' + str(loop))
                else:
                    logger.error('Loop file not found. Skipping loop: ' + str(loop))

                continue

            joined_sequence, bps, bp_cnt, stks, stk_cnt = load_loop_data(loop)
            node_features = []
            for nucl in joined_sequence:
                node_features.append(get_one_hot_encoded_nucl(nucl))

            # print(node_features)
            node_features = np.array(node_features)

            # print(joined_sequence, bps, stks)
            interaction_dict = {}

            for ind1 in bps:
                for ind2 in bps[ind1]:
                    # if len(bps[ind1][ind2]) > 1:
                        # print()
                        # print(bps[ind1][ind2])
                        # sys.exit()
                    for item in bps[ind1][ind2]:
                        i, j, nucl_pair, interaction = item
                        # interaction_dict[(i, j)] = get_encoded_interaction(interaction)
                        interaction_dict[(i, j)] = np.array(get_one_hot_encoded_interaction(interaction))

            for ind1 in stks:
                for ind2 in stks[ind1]:
                    for item in stks[ind1][ind2]:
                        i, j, nucl_pair, interaction = item
                        # interaction_dict[(i, j)] = get_encoded_interaction(interaction)
                        interaction_dict[(i, j)] = np.array(get_one_hot_encoded_interaction(interaction))

            edge_features = []
            for i in range(len(index_dict)):
                for j in range(len(index_dict)):
                    # if i == j:
                        # continue
                    if (i, j) in interaction_dict:
                        # edge_features.append([interaction_dict[(i, j)]])
                        edge_features.append(interaction_dict[(i, j)])
                    else:
                        # edge_features.append([-1])
                        edge_features.append(np.array(get_one_hot_encoded_interaction('')))

            edge_features = np.array(edge_features)


            # k = 0
            # for i in range(len(index_dict)):
            #     for j in range(len(index_dict)):
            #         print(i, j, edge_features[k])
            #         k += 1
            # print('node_features')
            # print(node_features)

            # print('adj_matrix')
            # print(adj_matrix)

            # print('edge_features')
            # print(edge_features)

            # print('encoded_fam_id')
            # print(encoded_fam_id)
            # sys.exit()
            # graph_list.append((known_motif_shortcode[family_id.lower()], Graph(x=node_features, a=adj_matrix, e=edge_features, y=encoded_fam_id)))
            
            # for supercluster_IL.in
            # graph_list.append((known_motif_shortcode[family_id.lower()], Graph(x=node_features, a=adj_matrix, e=edge_features, y=known_motif_shortcode[family_id.lower()])))

            # for all internal loops (IL_cluster_input.csv)
            graph_list.append((family_id, loop, Graph(x=node_features, a=adj_matrix, e=edge_features, y=family_id)))
            # graph_list.append(Graph(x=node_features, a=adj_matrix, y=encoded_fam_id))
            # print(graph_list)
            # sys.exit()
    # print(graph_list)
    # data = np.array(graph_list)
    # output_to_file(graph_list)
    motif_family_list = output_to_file_v2(user_input_fname, graph_list, output_dir, use_loop_name_in_graph_fname)
    return motif_family_list
    # global dataset
    # dataset = MyDataset(graph_list)

    # global number_of_labels
    # number_of_labels = len(families)

    # print(Dataset(graph_list))

def get_fam_id(loop, families):
    # default_val = get_loop_type(loop)
    for fam_id in families:
        if loop in families[fam_id]:
            if fam_id.lower() in known_motif_shortcode:
                return known_motif_shortcode[fam_id.lower()]
            # if fam_id.lower() in provided_family_info:
            #     return provided_family_info[fam_id.lower()]
            return fam_id
    # return default_val

def output_to_file_v2(user_input_fname, graph_list, graph_dir, use_loop_name_in_graph_fname):

    f = open(user_input_fname + ".pickle", 'rb')
    known_families = pickle.load(f)
    f.close()

    # provided_family_info = {}
    # if train_family_info == '1':
    #     f = open(os.path.join(data_dir, 'Train_family_info.csv'))
    #     lines = f.readlines()
    #     f.close()
    #     for line in lines[1:]:
    #         long_name, short_name = line.strip().split(',')
    #         provided_family_info[long_name.lower()] = short_name

    # if unknown_motif_family_info == '1':
    #     f = open(os.path.join(data_dir, 'Unknown_motif_family_info.csv'))
    #     lines = f.readlines()
    #     f.close()
    #     for line in lines[1:]:
    #         long_name, short_name = line.strip().split(',')
    #         provided_family_info[long_name.lower()] = short_name

    # graph_dir = os.path.join(data_dir, 'graphs_for_gnn_all_IL_ext5_in_coord')
    create_directory(graph_dir)

    graph_id = 0
    prev_fam_id = ''
    motif_family_list = []

    for i, (fam_id, loop, graph) in enumerate(graph_list):
        known_fam_id = get_fam_id(loop, known_families)
        #print(known_fam_id)
        if known_fam_id not in motif_family_list:
            motif_family_list.append(known_fam_id)

        if fam_id == prev_fam_id:
            graph_id += 1
        else:
            prev_fam_id = fam_id
            graph_id = 0
        # fp = open(os.path.join(graph_dir, fam_id + '_Graph_' + str(graph_id) + '.g'), 'w')
        graph_fname = known_fam_id + '_Graph_' + str(graph_id) + '.g'
        if use_loop_name_in_graph_fname:
            graph_fname = known_fam_id + '_' + loop.replace(':', '_') + '.g'
        # fp = open(os.path.join(graph_dir, known_fam_id + '_' + loop.replace(':', '_') + '.g'), 'w')
        # print(graph_fname)
        fp = open(os.path.join(graph_dir, graph_fname), 'w')
        fp.write('#Motif location:\n' + convert_a_loop_from_FASTA_to_PDB(loop) + '\t(PDB)\n' + loop + '\t(FASTA)\n\n')
        
        # if input_index_type == 'pdb':
        #     fp.write('(PDB)')
        # else:
        #     fp.write('(FASTA)')
        # fp.write('\n\n')
        fp.write('#Adjacency_Matrix:\n')
        for row in range(len(graph.a)):
            for col in range(len(graph.a[row])):
                fp.write(str(round(graph.a[row][col], 2)) + '\t')
            fp.write('\n')

        fp.write('#Node_Features:\n')
        for j in range(len(graph.x)):
            for k in range(4):
                fp.write(str(graph.x[j][k]) + '\t')
            fp.write('\n')

        fp.write('#Edge_Features:\n')
        for j in range(len(graph.e)):
            for k in range(len(graph.e[j])):
                fp.write(str(graph.e[j][k]) + '\t')
            # fp.write(str(graph.e[j]) + '\n')
            fp.write('\n')

        fp.write('#Graph_label:\n')
        fp.write(known_fam_id + '\n')
        fp.close()

    return motif_family_list

# def output_to_file(graph_list):

#     graph_dir = os.path.join(data_dir, 'graphs_for_gnn')
#     create_directory(graph_dir)

#     graph_id = 0
#     prev_fam_id = ''
#     for i, (fam_id, loop, graph) in enumerate(graph_list):
#         if fam_id == prev_fam_id:
#             graph_id += 1
#         else:
#             prev_fam_id = fam_id
#             graph_id = 0
#         fp = open(os.path.join(graph_dir, fam_id + '_Graph_' + str(graph_id) + '.g'), 'w')
#         fp.write('#Adjacency_Matrix:\n')
#         for row in range(len(graph.a)):
#             for col in range(len(graph.a[row])):
#                 fp.write(str(round(graph.a[row][col], 2)) + '\t')
#             fp.write('\n')

#         fp.write('#Node_Features:\n')
#         for j in range(len(graph.x)):
#             for k in range(4):
#                 fp.write(str(graph.x[j][k]) + '\t')
#             fp.write('\n')

#         fp.write('#Edge_Features:\n')
#         for j in range(len(graph.e)):
#             for k in range(len(graph.e[j])):
#                 fp.write(str(graph.e[j][k]) + '\t')
#             # fp.write(str(graph.e[j]) + '\n')
#             fp.write('\n')

#         fp.write('#Graph_label:\n')
#         fp.write(str(graph.y) + '\n')
#         fp.close()

if __name__ == '__main__':
    main()
