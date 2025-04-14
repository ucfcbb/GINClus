import os
import sys
import platform
from config import *

#from utils import strToNode, get_all_loop_combination, parse_scanx_alignment_block, parse_scanx_alignment_block_raw, parse_TM_alignment_block

from utils import parse_scanx_alignment_block, parse_scanx_alignment_block_raw
from utils import strToNode, parse_TM_alignment_block, convert_a_cluster_from_PDB_to_FASTA
from pymol_helper import get_pdb_index_list, get_atom_coordinate, aln_residue_temp, load_pdb_fasta_mapping_and_fasta_seq_dict
from superimposition_generator import extract_atom_coordinate, convert_array, kabsch_rmsd

#scripts_dir = '../..'
sys.path.append('../../')


tmalign_dir = os.path.join('src/my_lib/TMalign', 'TMalign-20180426')
loops_dir = 'data/loops'
tmalignment_dir = 'data/alignments/tmalign'
scanx_alignment_dir = 'data/alignments/scanx'



def rotate(l, x):
    return l[-x:] + l[:-x]

def get_all_loop_combination(loop):
    # print(loop)
    loop_combinations = []
    pdb_chain, regions = loop.strip().split(':')
    regions = regions.strip().split('_')
    loop = []
    for region in regions:
        s, e = region.strip().split('-')
        loop.append((s, e))
    
    for i in range(len(loop)):
        loop_i = rotate(loop, i)
        loop_combinations.append(pdb_chain + ':' + '_'.join(list(map(lambda x: '-'.join(x), loop_i))))

    # print(loop_combinations)
    return loop_combinations

def get_segment_count(loop):
    pdb_chain, regions = loop.strip().split(':')
    regions = regions.strip().split('_')
    return len(regions)

def generate_tm_alignment_file_for_single_loop(alignment_dir, l1, l2_list, partial_pdbx_dir):
    # print('tmalign alignment')
    partial_pdbx_dir0 = partial_pdbx_dir

    # file1 = os.path.join(loops_dir, l1 + '.smf')
    file1 = os.path.join(partial_pdbx_dir0, l1.replace(':', '_') + '.cif')
    output_fn = os.path.join(alignment_dir, l1.replace(':', '_') + '.aln')

    # scanx_aln_executable = os.path.join(motifscanx_dir, 'bin/align_ga')
    tmalign_executable = os.path.join(tmalign_dir, 'TMalign')
    # if platform.system() == 'Darwin':
    #   scanx_aln_executable = os.path.join(motifscanx_dir, 'bin/align_ga.mac')

    if not os.path.isfile(output_fn):
        print('Alignment file not present.')
        sys.exit()

    if not os.path.isfile(file1):
        print(l1.replace(':', '_') + ': Loop file not found.')
        sys.exit()

    for l2 in l2_list:
        if strToNode(l1) == strToNode(l2):
            continue
        # file2 = os.path.join(loop_dir, l2 + '.smf')
        file2 = os.path.join(partial_pdbx_dir0, l2.replace(':', '_') + '.cif')
        if not os.path.isfile(file2):
            print(l2.replace(':', '_') + ': Loop file not found.')
            sys.exit()

        # param_string = "--gap_o -5 --gap_e -3 --bp_q -3 --bp_t -3 --st_q -2 --st_t -2"
        param_string = '-infmt1 3 -infmt2 3 -outfmt -1 -a T -cp -mol RNA'
        l1_seg_count = get_segment_count(l1)
        l2_seg_count = get_segment_count(l2)
        if l1_seg_count == l2_seg_count and l1_seg_count != 2: # rotation will be tried with internal loops only
            param_string = '-infmt1 3 -infmt2 3 -outfmt -1 -a T -mol RNA'
        
        # if is_scanx_ga == False:
        #    param_string += " --SemiOriginal"

        #append
        # logger.info('Generating alignment for ' + l1 + ' and ' + l2)
        # os.system('%s %s %s %s >> %s' % (scanx_aln_executable, file1, file2, param_string, output_fn))
        os.system('%s %s %s %s >> %s' % (tmalign_executable, file1, file2, param_string, output_fn))

def create_directory(dir_to_create):
    if not os.path.exists(dir_to_create):
        os.makedirs(dir_to_create)

def preprocess_subcluster_output():

    temp_dir = 'output/temp'
    create_directory(temp_dir)

    input_fname = 'output/Subcluster_output.csv'
    output_fname = 'output/temp/Subcluster_output.in'
    fp = open(input_fname)
    lines = fp.readlines()
    fp.close()

    clusters = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        motif_id, cluster_no, subcluster_no, fam_id = line.strip().split('\t')
        cluster_id = 'subcluster_' + str(subcluster_no)
        pieces = motif_id.strip().split('_')
        # loop = pieces[0] + '_' + pieces[1] + '_' + pieces[2]
        loop = "_".join(x for x in pieces)
       
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(loop)

    fp = open(output_fname, 'w')
    for cluster_id in sorted(clusters):
        # print(cluster_id, end='')
        fp.write(cluster_id)
        for loop in clusters[cluster_id]:
            # print(loop)
            # print(',' + loop, end='')
            fp.write(',' + loop)
        # print('')
        fp.write('\n')
    fp.close()

    return len(clusters)

def generate_all_pair_values(clusters, partial_pdbx_dir, input_index_type):

    input_fname = 'output/temp/Subcluster_output.in'
    output_fname = 'output/temp/Subcluster_output_qscore_temp.csv'
   
    fp = open(input_fname)
    lines = fp.readlines()
    fp.close()

    clusters_all_comb = {}
    for cluster_id in clusters:
        clusters_all_comb[cluster_id] = []

        loops = clusters[cluster_id]
        for loop in loops:
            clusters_all_comb[cluster_id] += get_all_loop_combination(loop)

    loop_list = []

    for cluster_id in clusters:
        loops = clusters[cluster_id]
        loop_list.extend(loops)

    scanx_alignment_data = {}    
    tmalignment_data = {}

    # ScanX
    for cluster_id in clusters_all_comb:
        if cluster_id not in scanx_alignment_data:
            scanx_alignment_data[cluster_id] = {}
 
        loops = clusters_all_comb[cluster_id]
        
        for r1 in loops:
            fp = open(os.path.join(scanx_alignment_dir, r1.replace(':', '_') + '.aln'))
            lines = fp.readlines()
            fp.close()
            line_index = 0
            while line_index < len(lines):
                if lines[line_index].startswith('#  Aligning'):
                    test_r1, r2, cr1, cr2, aln1, aln2, score = parse_scanx_alignment_block(lines, line_index)
                    _, _, _, _, _, _, _, matching_bp_info, matching_stk_info, _, _, line_index = parse_scanx_alignment_block_raw(lines, line_index)
                    
                    if r1 != test_r1:
                        print('filename and loop mismatch. r1(filename): ' + r1 + ', r1(loop): ' + test_r1)
                        sys.exit()

                    if r1 not in scanx_alignment_data[cluster_id]:
                        scanx_alignment_data[cluster_id][r1] = {}
                    scanx_alignment_data[cluster_id][r1][r2] = (r1, r2, cr1, cr2, aln1, aln2, score, len(matching_bp_info), len(matching_stk_info))

                line_index += 1

    # TM-align
    for cluster_id in clusters:

        if cluster_id not in tmalignment_data:
            tmalignment_data[cluster_id] = {}

        # aln_dir = os.path.join(tmalignment_dir, cluster_id)

        loops = clusters[cluster_id]

        for r1 in loops:
            fname = os.path.join(tmalignment_dir, r1.replace(':', '_') + '.aln')
            if not os.path.exists(fname):
                print("Missing TMalign annotation: ", r1)
                continue

            fp = open(os.path.join(tmalignment_dir, r1.replace(':', '_') + '.aln'))
            lines = fp.readlines()
            fp.close()
            line_index = 0
            while line_index < len(lines):
                if lines[line_index].startswith('Name of Chain_1:'):
                    # test_r1, r2, aln1, aln2, TM_score, align_len, rmsd, seq_identity, tm_score_list = parse_TM_alignment_block(lines, line_index)
                    test_r1, r2, aln1, aln2, TM_score, align_len, rmsd, seq_identity, tm_score_list = parse_TM_alignment_block(lines, line_index)

                    if r1 != test_r1:
                        print('filename and loop mismatch. r1(filename): ' + r1 + ', r1(loop): ' + test_r1)
                        sys.exit()

                    if r1 not in tmalignment_data[cluster_id]:
                        #r1 = r1.replace(':', '_')
                        tmalignment_data[cluster_id][r1] = {}
                    tmalignment_data[cluster_id][r1][r2] = (r1, r2, aln1, aln2, TM_score, align_len, rmsd, seq_identity, tm_score_list)

                line_index += 1

    coord_dict = {}
    for lp in loop_list:
        lp = str(strToNode(lp))
        pdb_pm = get_pdb_index_list(lp)
        coord_backbone, coord_sugar, pdb_structure = get_atom_coordinate(os.path.join(partial_pdbx_dir, lp.replace(':', '_') + '.cif'), pdb_pm)
        coord_dict[lp] = (coord_backbone, coord_sugar)


    fp = open(output_fname, 'w')
    fp.write('Cluster_ID,Loop1,Loop2,ScanX_Score,ScanX_RMSD,AlignmentLength,M_BP,M_STK,TM-Score,TM_RMSD,AlignmentLength\n')
    fp.close()
    for cluster_id in clusters:
        loops = sorted(clusters[cluster_id])
 
        pdb_res_mapping_dict, fasta_seq_dict = load_pdb_fasta_mapping_and_fasta_seq_dict(cluster_id, scanx_alignment_data)
        for ii in range(len(loops)-1):
            r1 = loops[ii]
     
            for jj in range(ii+1,len(loops)):
                r2 = loops[jj]

                # Scanx
                if strToNode(r1) == strToNode(r2):
                    continue
                best_scanx_data = None
                for l1 in get_all_loop_combination(r1):
                    for l2 in get_all_loop_combination(r2):
                        try:
                            data = scanx_alignment_data[cluster_id][l1][l2]
                        except Exception as e:
                            print("Missing scanx alignment data: ", l1, l2)
                        if best_scanx_data == None:
                            best_scanx_data = data
                        else:
                            if float(data[6]) > float(best_scanx_data[6]):
                                best_scanx_data = data


                (l1, l2, cr1, cr2, aln1, aln2, score, matching_bp_cnt, matching_stk_cnt) = best_scanx_data
                pdb1_pm, pdb2_pm, i1_pm, i2_pm = aln_residue_temp(pdb_res_mapping_dict, fasta_seq_dict, l1, l2, cr1, cr2, aln1, aln2, 0, len(aln1)-1, 0)
                pdb_chain1, _ = r1.split(':')
                pdb_chain2, _ = r2.split(':')
                pdb1 = pdb_chain1.split('_')[0]
                pdb2 = pdb_chain2.split('_')[0]
                coord1 = extract_atom_coordinate(coord_dict[str(strToNode(l1))], pdb1_pm, pdb1)
                coord2 = extract_atom_coordinate(coord_dict[str(strToNode(l2))], pdb2_pm, pdb2)
                X, Y = convert_array(coord1, coord2)
                if len(X) != len(Y):
                    print('WARNING: Corresponding co-ordinates for alignments not found! rmsd = 20 assigned.')
                    rmsd = 20.
                elif len(X) == 0:
                    print('WARNING: Co-ordinates for alignments not found! rmsd = 20 assigned.')
                    rmsd = 20.
                else:
                    XC = sum(X)/len(X)
                    YC = sum(Y)/len(Y)
                    # calculating relative co-ordinate using mean as reference
                    X -= XC
                    Y -= YC
                    # time_s = time.time()
                    rmsd = kabsch_rmsd(X, Y)

                if r1 in tmalignment_data[cluster_id] and r2 in tmalignment_data[cluster_id][r1]:  
                    best_tmalign_data = tmalignment_data[cluster_id][r1][r2]
               
                else:
                    best_tmalign_data = (r1, r2, '', '', 0.0, 0, 10.0, 0.0, [])

                if r2 in tmalignment_data[cluster_id] and r1 in tmalignment_data[cluster_id][r2]:
                    if float(tmalignment_data[cluster_id][r2][r1][4]) > float(best_tmalign_data[4]):
                        best_tmalign_data = tmalignment_data[cluster_id][r2][r1]

                if strToNode(r1) in problematic_tmalign_loops or strToNode(r2) in problematic_tmalign_loops:
                    print("skipping alignment for pair: ", (r1, r2))
                    continue

                fp = open(output_fname, 'a')
                fp.write(cluster_id + ',' + r1 + ',' + r2 + ',')
                fp.write(str(best_scanx_data[6]) + ',' + str(rmsd) + ',' + str(min(len(best_scanx_data[4].replace('-','')), len(best_scanx_data[5].replace('-', '')))) + ',' + str(best_scanx_data[7]) + ',' + str(best_scanx_data[8]) + ',' + str(best_tmalign_data[4]) + ',' + str(best_tmalign_data[6]) + ',' + str(best_tmalign_data[5]) + '\n')
                fp.close()