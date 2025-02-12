import sys
import os
import platform
import glob
import logging
import operator
import multiprocessing as mp
import numpy
import time

# python 3 compatibility
from functools import reduce
from past.builtins import map

sys.path.append('../../')
from config import *
sys.path.append(scripts_dir)
from utils import *
from classes import *
from validators import *

# def read_existing_alignment_files():
# 	alignment_data = {}
# 	aln_file_list = glob.glob(os.path.join(alignment_dir, '*.aln'))
# 	for fn in aln_file_list:
# 		r1 = os.path.basename(fn)[:-4]
# 		node1 = strToNode(r1)

# 		if node1 not in alignment_data:
# 			alignment_data[node1] = {}

# 		fp = open(fn)
# 		lines = fp.readlines()
# 		fp.close()

# 		line_index = 0
# 		while line_index < len(lines):
# 			if lines[line_index].startswith('#  Aligning'):

# 				test_r1, r2, cr1, cr2, aln1, aln2, score = parse_scanx_alignment_block(lines, line_index)
# 				if r1 != test_r1:
# 					logger.error('filename and loop mismatch. r1(filename): ' + r1 + ', r1(loop): ' + test_r1)
# 					sys.exit()
				
# 				node2 = strToNode(r2)
# 				if node2 not in alignment_data:
# 					alignment_data[node2] = {}

# 				alignment_data[node1][node2] = (r1, r2, 0.0, cr1, cr2, aln1, aln2, score)	#dummy zscore, remove and adjust
# 				alignment_data[node2][node1] = (r2, r1, 0.0, cr2, cr1, aln2, aln1, score)	#dummy zscore, remove and adjust

# 			line_index += 1

# 	return alignment_data

def generate_tm_alignment_file_for_single_loop(alignment_dir, l1, l2_list):
	partial_pdbx_dir0 = os.path.join(data_dir, 'pdbx_extracted_ext0')

	# file1 = os.path.join(loop_dir, l1 + '.smf')
	file1 = os.path.join(partial_pdbx_dir0, l1.replace(':', '_') + '.cif')
	output_fn = os.path.join(alignment_dir, l1.replace(':', '_') + '.aln')

	# scanx_aln_executable = os.path.join(motifscanx_dir, 'bin/align_ga')
	tmalign_executable = os.path.join(tmalign_dir, 'TMalign')
	# if platform.system() == 'Darwin':
	# 	scanx_aln_executable = os.path.join(motifscanx_dir, 'bin/align_ga.mac')

	if not os.path.isfile(output_fn):
		logger.error('Alignment file not present.')
		sys.exit()

	if not os.path.isfile(file1):
		logger.error(l1 + ': Loop file not found.')
		sys.exit()

	for l2 in l2_list:
		if strToNode(l1) == strToNode(l2):
			continue
		# file2 = os.path.join(loop_dir, l2 + '.smf')
		file2 = os.path.join(partial_pdbx_dir0, l2.replace(':', '_') + '.cif')
		if not os.path.isfile(file2):
			logger.error(l2 + ': Loop file not found.')
			sys.exit()

		# param_string = "--gap_o -5 --gap_e -3 --bp_q -3 --bp_t -3 --st_q -2 --st_t -2"
		param_string = '-infmt1 3 -infmt2 3 -outfmt -1 -a T -cp -mol RNA'
		l1_seg_count = get_segment_count(l1)
		l2_seg_count = get_segment_count(l2)
		if l1_seg_count == l2_seg_count and l1_seg_count != 2: # rotation will be tried with internal loops only
			param_string = '-infmt1 3 -infmt2 3 -outfmt -1 -a T -mol RNA'
		# if is_scanx_ga == False:
		#	 param_string += " --SemiOriginal"

		#append
		# logger.info('Generating alignment for ' + l1 + ' and ' + l2)
		# os.system('%s %s %s %s >> %s' % (scanx_aln_executable, file1, file2, param_string, output_fn))
		os.system('%s %s %s %s >> %s' % (tmalign_executable, file1, file2, param_string, output_fn))

def generate_alignment_file_for_single_loop(alignment_dir, l1, l2_list):

	if strToNode(l1) in problematic_scanx_alignments:
		return

	file1 = os.path.join(loop_dir, l1.replace(':', '_') + '.smf')
	output_fn = os.path.join(alignment_dir, l1.replace(':', '_') + '.aln')

	scanx_aln_executable = os.path.join(motifscanx_dir, 'bin/align_ga')
	if platform.system() == 'Darwin':
		scanx_aln_executable = os.path.join(motifscanx_dir, 'bin/align_ga.mac')

	if not os.path.isfile(output_fn):
		logger.error('Alignment file not present.')
		sys.exit()

	if not os.path.isfile(file1):
		logger.error(l1 + ': Loop file not found.')
		sys.exit()

	for l2 in l2_list:
		if strToNode(l1) == strToNode(l2):
			continue
		file2 = os.path.join(loop_dir, l2.replace(':', '_') + '.smf')
		if not os.path.isfile(file2):
			logger.error(l2 + ': Loop file not found.')
			sys.exit()

		param_string = "--gap_o -5 --gap_e -3 --bp_q -3 --bp_t -3 --st_q -2 --st_t -2"
		# if is_scanx_ga == False:
		#	 param_string += " --SemiOriginal"

		#append
		# logger.info('Generating alignment for ' + l1 + ' and ' + l2)
		exit_code = os.system('%s %s %s %s >> %s' % (scanx_aln_executable, file1, file2, param_string, output_fn))
		# print(exit_code)
		if exit_code == 35584:
			filex = os.path.join(motifscanx_dir, 'models/k-turn_consensus.struct')
			outputx = os.path.join(root_dir, 'output/temp/temp.txt')
			if strToNode(l1) not in problematic_scanx_alignments:
				ec = os.system('%s %s %s %s > %s' % (scanx_aln_executable, file1, filex, param_string, outputx))
				if ec == 35584:
					problematic_scanx_alignments.append(strToNode(l1))
					
			if strToNode(l2) not in problematic_scanx_alignments:
				ec = os.system('%s %s %s %s > %s' % (scanx_aln_executable, file2, filex, param_string, outputx))
				if ec == 35584:
					problematic_scanx_alignments.append(strToNode(l2))

# def _generate_alignment_files_worker(p):
# 	generate_alignment_file_for_single_loop(*p)

def create_empty_alignment_files(alignment_dir, loops):
	for loop in loops:
		loop_fn = os.path.join(alignment_dir, loop.replace(':', '_') + '.aln')
		if not os.path.isfile(loop_fn):
			fp = open(loop_fn, 'w')
			fp.close()

# def write_alignment_for_other_pairs(alignment_dir, loops, existing_alignment_data, rewritten_list=None):
# 	alignment_to_write = {}
	
# 	for i in range(len(loops)):
# 		r1 = loops[i]
# 		if rewritten_list != None:
# 			if r1 in rewritten_list:
# 				continue
# 			rewritten_list.append(r1)

# 		fp = open(os.path.join(alignment_dir, r1 + '.aln'))
# 		lines = fp.readlines()
# 		fp.close()
# 		line_index = 0
# 		while line_index < len(lines):
# 			if lines[line_index].startswith('#  Aligning'):
# 				test_r1, r2, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied = parse_scanx_alignment_block_raw(lines, line_index)
# 				if is_copied == True:
# 					break
# 				if r1 != test_r1:
# 					logger.error('filename and loop mismatch. r1(filename): ' + r1 + ', r1(loop): ' + test_r1)
# 					sys.exit()

# 				if r2 not in alignment_to_write:
# 					alignment_to_write[r2] = []

# 				if r2 not in existing_alignment_data:
# 					alignment_to_write[r2].append((r1, r2, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied))
# 					# cnt += 1
# 				else:
# 					if r1 not in existing_alignment_data[r2]:
# 						alignment_to_write[r2].append((r1, r2, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied))
# 						# cnt += 1

# 			line_index += 1

# 	for r2 in alignment_to_write:
# 		for r1, _, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied in alignment_to_write[r2]:
# 			fp = open(os.path.join(alignment_dir, r2 + '.aln'), 'a')
# 			fp.write('\n#  Aligning       :: ' + r2 + ' and ' + r1 + ':\n')
# 			fp.write('#  Alignment score:  ' + str(score) + '\n')
# 			fp.write('#  P-value:  NA\n')
# 			fp.write('#  Query aligned region: ' + cr2 + '\n')
# 			fp.write('#  Target aligned region: ' + cr1 + '\n\n')
# 			fp.write('\t' + aln2 + '\n')
# 			fp.write('\t' + aln1 + '\n\n')
# 			fp.write('#  Matched base-pairing interactions: \n')
# 			for item in matching_bp_info:
# 				fp.write('\t' + item[1] + ' ' * (47 - len(item[1])) + 'MATCHES\t' + item[0] + '\n')
# 			fp.write('#  Matched base-stacking interactions: \n')
# 			for item in matching_stk_info:
# 				fp.write('\t' + item[1] + ' ' * (47 - len(item[1])) + 'MATCHES\t' + item[0] + '\n')
# 			fp.write('Total Elapsed Time : ' + str(elapsed_time) + ' sec (copied)\n')

# 			fp.close()

def write_alignment_for_other_pairs(alignment_dir, loops, existing_alignment_data, missing_reverse_pair_aln_dict):
	# rewritten_list = []
	alignment_to_write = {}
	
	for i in range(len(loops)):
		r1 = loops[i]
		# if r1 in rewritten_list:
		# 	continue
		# rewritten_list.append(r1)

		fp = open(os.path.join(alignment_dir, r1.replace(':', '_') + '.aln'))
		lines = fp.readlines()
		fp.close()
		line_index = 0
		while line_index < len(lines):
			if lines[line_index].startswith('#  Aligning'):
				test_r1, r2, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied, line_index = parse_scanx_alignment_block_raw(lines, line_index)

				if is_copied == True:
					line_index += 1
					continue

				if r1 != test_r1:
					logger.error('filename and loop mismatch. r1(filename): ' + r1 + ', r1(loop): ' + test_r1)
					sys.exit()

				if r2 in missing_reverse_pair_aln_dict:
					if r1 in missing_reverse_pair_aln_dict[r2]:

						if r2 not in alignment_to_write:
							alignment_to_write[r2] = []

						alignment_to_write[r2].append((r1, r2, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied))

			line_index += 1

	for r2 in alignment_to_write:
		for r1, _, cr1, cr2, aln1, aln2, score, matching_bp_info, matching_stk_info, elapsed_time, is_copied in alignment_to_write[r2]:
			fp = open(os.path.join(alignment_dir, r2.replace(':', '_') + '.aln'), 'a')
			fp.write('\n#  Aligning       :: ' + r2 + ' and ' + r1 + ':\n')
			fp.write('#  Alignment score:  ' + str(score) + '\n')
			fp.write('#  P-value:  NA\n')
			fp.write('#  Query aligned region: ' + cr2 + '\n')
			fp.write('#  Target aligned region: ' + cr1 + '\n\n')
			fp.write('\t' + aln2 + '\n')
			fp.write('\t' + aln1 + '\n\n')
			fp.write('#  Matched base-pairing interactions: \n')
			for item in matching_bp_info:
				fp.write('\t' + item[1] + ' ' * (47 - len(item[1])) + 'MATCHES\t' + item[0] + '\n')
			fp.write('#  Matched base-stacking interactions: \n')
			for item in matching_stk_info:
				fp.write('\t' + item[1] + ' ' * (47 - len(item[1])) + 'MATCHES\t' + item[0] + '\n')
			fp.write('Total Elapsed Time : ' + str(elapsed_time) + ' sec (copied)\n')

			fp.close()

# def get_mp_alignment_param_list(alignment_dir, loops, existing_alignment_data):
# 	param_list = []
# 	create_empty_alignment_files(alignment_dir, loops)
# 	for i in range(len(loops)):
# 		l1 = loops[i]
# 		# node1 = strToNode(l1)
# 		start = i + 1	# n * (n + 1) / 2 in place of n * n
# 		if l1 not in existing_alignment_data:
# 			param_list.append((l1, loops[start:]))
# 		else:
# 			# print('some alignment exists, skip them')
# 			# some alignment exists, skip them
# 			new_list = []
# 			for lp in loops[start:]:
# 				if lp not in existing_alignment_data[l1]:
# 					new_list.append(lp)
# 			param_list.append((l1, new_list))

# 	return param_list

def get_mp_alignment_param_list(alignment_dir, alignment_tool, loops, existing_alignment_data):
	param_list = []
	missing_reverse_pair_aln_dict = {}
	create_empty_alignment_files(alignment_dir, loops)

	# if alignment_tool == 'tmalign':
	# 	for i in range(len(loops)):
	# 		l1 = loops[i]
	# 		l1_aln_list = []
	# 		for j in range(len(loops)):
	# 			l2 = loops[j]
	# 			if i != j:
	# 				l1_aln_list.append(l2)
			
	# 		if len(l1_aln_list) > 0:
	# 			param_list.append((l1, l1_aln_list))

	# 	return param_list, missing_reverse_pair_aln_dict

	for i in range(len(loops)):
		l1 = loops[i]
		l1_aln_list = []

		start = i+1

		if alignment_tool == 'tmalign':
			start = 0

		for j in range(start, len(loops)):

			if i == j:
				continue

			l2 = loops[j]
			has_l1_l2 = l1 in existing_alignment_data and l2 in existing_alignment_data[l1]
			has_l2_l1 = l2 in existing_alignment_data and l1 in existing_alignment_data[l2]
			
			if has_l1_l2 == False and has_l2_l1 == False:
				l1_aln_list.append(l2)
				if l2 not in missing_reverse_pair_aln_dict:
					missing_reverse_pair_aln_dict[l2] = []
				missing_reverse_pair_aln_dict[l2].append(l1)
			
			elif has_l1_l2 == False:
				if l1 not in missing_reverse_pair_aln_dict:
					missing_reverse_pair_aln_dict[l1] = []
				missing_reverse_pair_aln_dict[l1].append(l2)

			elif has_l2_l1 == False:
				if l2 not in missing_reverse_pair_aln_dict:
					missing_reverse_pair_aln_dict[l2] = []
				missing_reverse_pair_aln_dict[l2].append(l1)
		
		if len(l1_aln_list) > 0:
			param_list.append((l1, l1_aln_list))

	return param_list, missing_reverse_pair_aln_dict

def get_existing_TMalignment_info(alignment_dir, loop_node_list_str):
	loop_node_list = map(lambda x: strToNode(x), loop_node_list_str)

	existing_alignment_data = {}
	is_valid = True
	for node_str in loop_node_list_str:
		# for loop in get_all_loop_combination(node_str):
		for loop in [node_str]:
			fn = os.path.join(alignment_dir, loop.replace(':', '_') + '.aln')
			if not os.path.isfile(fn):
				continue

	# for fn in glob.glob(os.path.join(alignment_dir, '*.aln')):
			fp = open(fn)
			lines = fp.readlines()
			fp.close()			
			existing_alignment_data[loop] = []

			line_index = 0
			while line_index < len(lines):
			# for line in lines:
				line = lines[line_index]
				# if line.startswith('#  Aligning       :: '):
				if line.startswith('Name of Chain_1:'):
					# print(line)
					l1_line = line
					l2_line = lines[line_index+1]

					l1 = os.path.basename(l1_line.strip().split(' ')[3].strip())[:-4]
					l2 = os.path.basename(l2_line.strip().split(' ')[3].strip())[:-4]

					l1 = '_'.join(l1.strip().split('_')[:2]) + ':' + '_'.join(l1.strip().split('_')[2:])
					l2 = '_'.join(l2.strip().split('_')[:2]) + ':' + '_'.join(l2.strip().split('_')[2:])

					if l1 != loop:
						print('l1 mismatch with filename in existing alignments. Invalidating existing alignments.')
						is_valid = False
						# sys.exit()
					if strToNode(l2) not in loop_node_list:
						continue

					existing_alignment_data[l1].append(l2)
					# line_index += 1

				line_index += 1

	if is_valid == False:
		existing_alignment_data = {}	# Invalidating all existing alignments. System will regenerate all alignments.

	return existing_alignment_data
	
def get_existing_alignment_info(alignment_dir, loop_node_list_str):
	loop_node_list = map(lambda x: strToNode(x), loop_node_list_str)

	existing_alignment_data = {}
	is_valid = True
	for node_str in loop_node_list_str:
		for loop in get_all_loop_combination(node_str):
			fn = os.path.join(alignment_dir, loop.replace(':', '_') + '.aln')
			if not os.path.isfile(fn):
				continue

	# for fn in glob.glob(os.path.join(alignment_dir, '*.aln')):
			fp = open(fn)
			lines = fp.readlines()
			fp.close()			
			existing_alignment_data[loop] = []

			for line in lines:
				if line.startswith('#  Aligning       :: '):
					# print(line)
					l1, l2 = line.strip().split('::')[1].strip()[:-1].split('and')
					l1 = l1.strip()
					l2 = l2.strip()
					# print(l1, l2)
					# sys.exit()
					if l1 != loop:
						print('l1 mismatch with filename in existing alignments. Invalidating existing alignments.')
						is_valid = False
						# sys.exit()
					if strToNode(l2) not in loop_node_list:
						continue

					existing_alignment_data[l1].append(l2)

	if is_valid == False:
		existing_alignment_data = {}	# Invalidating all existing alignments. System will regenerate all alignments.

	return existing_alignment_data

def merge_list_dict(missing_reverse_pair_aln_dict, missing_pair_dict):
	for key in missing_pair_dict:
		if key not in missing_reverse_pair_aln_dict:
			missing_reverse_pair_aln_dict[key] = []
		missing_reverse_pair_aln_dict[key] += missing_pair_dict[key]

def get_alignment_files(alignment_dir, alignment_tool, families, loop_node_list_str, is_alignment_from_user):
	create_directory(alignment_dir)

	# if is_alignment_from_user == False:
	# 	logger.info('Alignment files not provided by user.')
	# else:
	# 	logger.info('Alignment files provided by user.')

	logger.info('Generating alignment data from ' + alignment_dir[base_path_len:] + '.')
	# logger.info('Loading existing alignment data from ' + alignment_dir[base_path_len:] + '.')

	if alignment_tool == 'tmalign':
		existing_alignment_data = get_existing_TMalignment_info(alignment_dir, loop_node_list_str)
	else:
		existing_alignment_data = get_existing_alignment_info(alignment_dir, loop_node_list_str)

	start_time = time.time()
	
	parameter_list = []
	missing_reverse_pair_aln_dict = {}

	loops = []
	loops_list_of_list = []
	if alignment_tool == 'tmalign':
		for family in families:
			# family_loops = reduce(operator.concat, list(map(lambda x: get_all_loop_combination(x), families[family])))
			family_loops = []
			for x in families[family]:
				# family_loops += get_all_loop_combination(x)
				family_loops.append(str(strToNode(x)))
			loops += family_loops
			loops_list_of_list.append(family_loops)
	else:
		for family in families:
			# family_loops = reduce(operator.concat, list(map(lambda x: get_all_loop_combination(x), families[family])))
			family_loops = []
			for x in families[family]:
				family_loops += get_all_loop_combination(x)
				# family_loops.append(str(strToNode(x)))
			loops += family_loops
			loops_list_of_list.append(family_loops)

	loops = list(set(loops))	#removing duplicates (if any)

	if align_all_pair == True:
		parameter_list, missing_reverse_pair_aln_dict = get_mp_alignment_param_list(alignment_dir, alignment_tool, loops, existing_alignment_data)
	else:
		for lps in loops_list_of_list:
			p_list, missing_pair_dict = get_mp_alignment_param_list(alignment_dir, alignment_tool, lps, existing_alignment_data)
			parameter_list += p_list
			merge_list_dict(missing_reverse_pair_aln_dict, missing_pair_dict)

		# removing duplicate loop from different families
		for key in missing_reverse_pair_aln_dict:
			missing_reverse_pair_aln_dict[key] = list(set(missing_reverse_pair_aln_dict[key]))

	# print(parameter_list)
	# sys.exit()

	# if len(parameter_list) > 0:
	# 	if alignment_tool == 'tmalign':
	# 		logger.info('All required alignments not found, generating using TMalign.')
	# 	else:
	# 		logger.info('All required alignments not found, generating using ScanX.')

	if alignment_tool != 'tmalign':
		os.environ['RNAMOTIFSCANX_PATH'] = motifscanx_dir
		os.environ['RNAVIEW'] = os.path.join(motifscanx_dir, 'thirdparty/RNAVIEW')

	# pool = mp.Pool(number_of_multiprocess)
	# pool.map(_generate_alignment_files_worker, parameter_list)

	for l1, l2_list in parameter_list:
		if alignment_tool == 'tmalign':
			generate_tm_alignment_file_for_single_loop(alignment_dir, l1, l2_list)
		else:
			generate_alignment_file_for_single_loop(alignment_dir, l1, l2_list)

	# check and write alignment for other pairs
	# if align_all_pair == True:
	# 	write_alignment_for_other_pairs(alignment_dir, loops, existing_alignment_data)
	# else:
	# 	rewritten_list = []
	# 	for lps in loops_list_of_list:
	# 		# print(rewritten_list)
	# 		write_alignment_for_other_pairs(alignment_dir, lps, existing_alignment_data, rewritten_list)
			# print(rewritten_list)

	if len(missing_reverse_pair_aln_dict) > 0 and alignment_tool == 'scanx':
		# if output_env == 'local':
		# 	logger.info('Writing alignments for reverse pairs.')

		write_alignment_for_other_pairs(alignment_dir, loops, existing_alignment_data, missing_reverse_pair_aln_dict)

	# logger.info('Alignment preparation done.')
	# logger.info('Time taken: ' + str(round((time.time() - start_time), 3)) + ' seconds.\n')

def is_r1_r2_in_same_family(r1_corresponding_list, r2):
	for loops in r1_corresponding_list:
		if r2 in loops:
			return True
	return False

def get_r1_corresponding_list(loops_list_of_list, r1):
	corresponding_list = []
	for loops in loops_list_of_list:
		if r1 in loops:
			corresponding_list.append(loops)
	return corresponding_list

def get_normalized_data_for_TMalign(alignment_dir, families):
	score_dict = {}
	query_score_dict = {}

	loops = []
	loops_list_of_list = []
	for family in families:
		family_loops = []
		for lp in families[family]:
			# family_loops += get_all_loop_combination(lp)
			family_loops.append(str(strToNode(lp)))
		loops += family_loops
		loops_list_of_list.append(family_loops)
	loops = list(set(loops))	#removing duplicates (if any)

	for r1 in loops:
		alignment_fname = os.path.join(alignment_dir, r1.replace(':', '_') + '.aln')
		# print(alignment_fname)

		if not os.path.isfile(alignment_fname):
			return None
		
		r1_corresponding_list = [loops]
		if align_all_pair == False:
			r1_corresponding_list = get_r1_corresponding_list(loops_list_of_list, r1)

		if r1 not in query_score_dict:
			query_score_dict[r1] = []

		fp = open(alignment_fname)
		lines = fp.readlines()
		fp.close()

		line_index = 0
		# r1, r2, aln1, aln2, TM_score, align_len, rmsd, seq_identity
		while line_index < len(lines):
			line = lines[line_index]

			if line.startswith('Name of Chain_1:'):
				# r1_line = line
				r2_line = lines[line_index+1]
				score_line = lines[line_index+8]

				# r1 = os.path.basename(r1_line.strip().split(' ')[3].strip())[:-4]
				r2 = os.path.basename(r2_line.strip().split(' ')[3].strip())[:-4]

				# r1 = r1.strip()
				# r2 = r2.strip()
				r2 = '_'.join(r2.strip().split('_')[:2]) + ':' + '_'.join(r2.strip().split('_')[2:])
				if strToNode(r1) == strToNode(r2):
					line_index += 1
					continue
				if is_r1_r2_in_same_family(r1_corresponding_list, r2) == False:
					line_index += 1
					continue

				score_text = score_line.strip().split('(')[0].strip().split('=')[1].strip()
				if score_text == '':
					score = -50.
				else:
					score = float(score_text)

				if (r1, r2) in score_dict or (r2, r1) in score_dict:
					line_index += 1
					continue

				score_dict[(r1, r2)] = score
				score_dict[(r2, r1)] = score

				if r2 not in query_score_dict:
					query_score_dict[r2] = []
				query_score_dict[r1].append(score)
				query_score_dict[r2].append(score)

			line_index += 1

	param = {}
	for r in query_score_dict:
		# print('calculating mean of ' + str(len(query_score_dict[r])) + ' elements')
		m = numpy.mean(query_score_dict[r])
		std = numpy.std(query_score_dict[r])
		param[r] = (float(m), float(std))

	normalized_score_dict = {}
	for r1, r2 in score_dict:
		#print r1, r2, -1.*zscore(score_dict[(r1, r2)], param[r1][0], param[r1][1])
		normalized_score_dict[(r1, r2)] = zscore(score_dict[(r1, r2)], param[r1][0], param[r1][1])
	
	return normalized_score_dict

def get_normalized_data(alignment_dir, families):
	score_dict = {}
	query_score_dict = {}

	loops = []
	loops_list_of_list = []
	for family in families:
		family_loops = []
		for lp in families[family]:
			family_loops += get_all_loop_combination(lp)
		loops += family_loops
		loops_list_of_list.append(family_loops)
	loops = list(set(loops))	#removing duplicates (if any)

	for r1 in loops:
		alignment_fname = os.path.join(alignment_dir, r1.replace(':', '_') + '.aln')

		if not os.path.isfile(alignment_fname):
			return None
		
		r1_corresponding_list = [loops]
		if align_all_pair == False:
			r1_corresponding_list = get_r1_corresponding_list(loops_list_of_list, r1)

		if r1 not in query_score_dict:
			query_score_dict[r1] = []

		fp = open(alignment_fname)
		flag = 0
		for line in fp.readlines():
			if line.startswith("#  Aligning") and flag == 0:
				r2 = line.split("::")[1].split(" and ")[1].strip().strip(":")

				if strToNode(r1) == strToNode(r2):
					continue
				
				if is_r1_r2_in_same_family(r1_corresponding_list, r2) == False:
					continue

				flag = 1

			elif line.startswith("#  Alignment score:") and flag == 1 :
				flag = 0
				score_text = line.split(":")[1].strip()
				if score_text == '':
					score = -50.
				else:
					score = float(score_text)

				# print "r1: " + str(r1) + ", r2: " + str(r2) + ", score: " + str(score)
				if (r1, r2) in score_dict or (r2, r1) in score_dict:
					continue

				score_dict[(r1, r2)] = score
				score_dict[(r2, r1)] = score

				if r2 not in query_score_dict:
					query_score_dict[r2] = []
				query_score_dict[r1].append(score)
				query_score_dict[r2].append(score)

	param = {}
	for r in query_score_dict:
		m = numpy.mean(query_score_dict[r])
		std = numpy.std(query_score_dict[r])
		param[r] = (float(m), float(std))

	normalized_score_dict = {}
	for r1, r2 in score_dict:
		normalized_score_dict[(r1, r2)] = zscore(score_dict[(r1, r2)], param[r1][0], param[r1][1])
	
	return normalized_score_dict

def delete_graph_file(input_fname_base, graphs_and_pickles_dir, alignment_tool):
	fname = os.path.join(graphs_and_pickles_dir, input_fname_base + '_' + alignment_tool + '.z.graph')
	if os.path.isfile(fname):
		os.remove(fname)

def generate_best_alignment_data(input_fname_base, graphs_and_pickles_dir, alignment_dir, alignment_tool, families):

	fname = os.path.join(graphs_and_pickles_dir, input_fname_base + '_' + alignment_tool + '.z.graph')

	if is_valid_graph(families, fname) == True:
		logger.info('Using existing graph file from previous run (In ' + fname[base_path_len:] + ').')
		print('')
		# return True, True 	# previous_graph_file_reused = True, all_alignment_files_found = True
		return True 	# previous_graph_file_reused = True, all_alignment_files_found = True

	# logger.info('Processing alignment data ...')
	# start_time = time.time()

	if alignment_tool == 'tmalign':
		alignment_score = get_normalized_data_for_TMalign(alignment_dir, families)
	else:
		alignment_score = get_normalized_data(alignment_dir, families)		# actually z-score

	if alignment_score == None:
		# return False, False # previous_graph_file_reused = False, all_alignment_files_found = False
		return False

	fw = open(fname, 'w')

	edge_score_dict = {}
	checked_edge = set([])
	edge_dict = {}

	# print_a_dict_sorted(alignment_score)
	# print('')
	for l1, l2 in sorted(alignment_score):
		if (l1, l2) in checked_edge:
			continue

		# print(l1, l2)
		# (l2, l1) will not be checked again
		checked_edge.add((l1, l2))
		checked_edge.add((l2, l1))

		# if (l2, l1) in alignment_score:
		# alignment score is same but z-score may be different
		# pick the max z-score
		edge_score = max(alignment_score[(l1, l2)], alignment_score[(l2, l1)])
		if isClose(edge_score, alignment_score[(l1, l2)], 0.0000001):
			t1 = l1
			t2 = l2
		else:
			t1 = l2
			t2 = l1

		# (chain1, loop1) = l1.split(':')
		# (chain2, loop2) = l2.split(':')

		# region1 = loop1.split('_')
		# region2 = loop2.split('_')

		# node1 = Node(chain1, region1)
		# node2 = Node(chain2, region2)

		node1 = strToNode(l1)
		node2 = strToNode(l2)

		if node1 != node2:
			edge = Edge(node1, node2)

			if edge not in edge_dict:
				edge_dict[edge] = (edge_score, t1, t2)
			else:
				best_edge_score, best_t1, best_t2 = edge_dict[edge]
				if (best_edge_score < edge_score):
					if isClose(edge_score, best_edge_score, 0.0000001):
						if t1 < best_t1 or (t1 == best_t1 and t2 < best_t2):
							edge_dict[edge] = (edge_score, t1, t2)
					else:
						edge_dict[edge] = (edge_score, t1, t2)

			# if (edge not in edge_dict) or (edge_dict[edge][0] < edge_score):
			# 	edge_dict[edge] = (edge_score, t1, t2)

	for edge in edge_dict:
		edge_score_dict[(edge.node1, edge.node2)] = edge_dict[edge][0]
		fw.write(str(edge.node1) + ' ' + str(edge.node2) + ' ' + str(edge_dict[edge][0])+ ' ' + edge_dict[edge][1] + ' ' + edge_dict[edge][2] + '\n')
	fw.close()
	
	# logger.info('Done')
	# logger.info('Time taken: ' + str(round((time.time() - start_time), 3)) + ' seconds.\n')

	# return False, True 	# previous_graph_file_reused = False, all_alignment_files_found = True
	return True

def adjust_rotated_tmalign_alignment_result_v2(loop1, loop2, aln1, aln2):
	# this function is intended to work with Internal Loops only
	if '*' in aln1:
		star_ind = aln1.index('*')
		loop1_part1 = aln1[:star_ind].replace('-', '')
		loop1_part2 = aln1[star_ind+1:].replace('-', '')
		loop2_part1 = aln2[:star_ind].replace('-', '')
		loop2_part2 = aln2[star_ind+1:].replace('-', '')

		if len(loop2_part2) == 0 or len(loop2_part1) == 0:
			aln1 = aln1[star_ind+1:] + aln1[:star_ind]
			aln2 = aln2[star_ind+1:] + aln2[:star_ind]

		# elif len(loop2_part1) == 0:
		# 	aln1 = aln1[star_ind+1:] + aln1[:star_ind]
		# 	aln2 = aln2[star_ind+1:] + aln2[:star_ind]

	return aln1, aln2

def adjust_rotated_tmalign_alignment_result(loop1, loop2, aln1, aln2):
	# this function is intended to work with Internal Loops only
	# print(loop1)
	# print(loop2)
	if '*' in aln1:
		star_ind = aln1.index('*')
		loop1_part1 = aln1[:star_ind].replace('-', '')
		loop1_part2 = aln1[star_ind+1:].replace('-', '')
		loop2_part1 = aln2[:star_ind].replace('-', '')
		loop2_part2 = aln2[star_ind+1:].replace('-', '')

		loop1_info = load_loop_data(loop1)
		loop2_info = load_loop_data(loop2)

		sequence1, _, _, _, _ = loop1_info
		sequence2, _, _, _, _ = loop2_info
		loop1_parts = get_separated_sequences(loop1, sequence1)
		loop2_parts = get_separated_sequences(loop2, sequence2)

		if (loop1_part1 != loop1_parts[1] and loop1_part2 != loop1_parts[0]):
			if len(loop1_part1) > len(loop1_parts[1]):
				while len(loop1_part1) > len(loop1_parts[1]) and (aln1[0] != '-' and aln2[0] == '-'):
					aln1 = rotate_left(aln1)
					aln2 = rotate_left(aln2)
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')

				star_ind = aln1.index('*')
				while len(loop1_part1) > len(loop1_parts[1]) and (aln1[star_ind-1] != '-' and aln2[star_ind-1] == '-'):
					aln1 = star_move_left(aln1)
					aln2 = star_move_left(aln2[:star_ind] + '*' + aln2[star_ind+1:])
					aln2 = aln2.replace('*', '-')
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')
			else:
				while len(loop1_part1) < len(loop1_parts[1]) and (aln1[len(aln1)-1] != '-' and aln2[len(aln2)-1] == '-'):
					aln1 = rotate_right(aln1)
					aln2 = rotate_right(aln2)
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')

				star_ind = aln1.index('*')
				while len(loop1_part1) < len(loop1_parts[1]) and (aln1[star_ind+1] != '-' and aln2[star_ind+1] == '-'):
					aln1 = star_move_right(aln1)
					aln2 = star_move_right(aln2[:star_ind] + '*' + aln2[star_ind+1:])
					aln2 = aln2.replace('*', '-')
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')

		elif (loop2_part1 != loop2_parts[0] and loop2_part2 != loop2_parts[1]):
			if len(loop2_part1) > len(loop2_parts[0]):
				while len(loop2_part1) > len(loop2_parts[0]) and (aln1[0] == '-' and aln2[0] != '-'):
					aln1 = rotate_left(aln1)
					aln2 = rotate_left(aln2)
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')

				star_ind = aln1.index('*')
				while len(loop2_part1) > len(loop2_parts[0]) and (aln1[star_ind-1] == '-' and aln2[star_ind-1] != '-'):
					aln1 = star_move_left(aln1)
					aln2 = star_move_left(aln2[:star_ind] + '*' + aln2[star_ind+1:])
					aln2 = aln2.replace('*', '-')
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')
			else:
				while len(loop2_part1) < len(loop2_parts[0]) and (aln1[len(aln1)-1] == '-' and aln2[len(aln2)-1] != '-'):
					aln1 = rotate_right(aln1)
					aln2 = rotate_right(aln2)
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')

				star_ind = aln1.index('*')
				while len(loop2_part1) < len(loop2_parts[0]) and (aln1[star_ind+1] == '-' and aln2[star_ind+1] != '-'):
					aln1 = star_move_right(aln1)
					aln2 = star_move_right(aln2[:star_ind] + '*' + aln2[star_ind+1:])
					aln2 = aln2.replace('*', '-')
					star_ind = aln1.index('*')
					loop1_part1 = aln1[:star_ind].replace('-', '')
					loop1_part2 = aln1[star_ind+1:].replace('-', '')
					loop2_part1 = aln2[:star_ind].replace('-', '')
					loop2_part2 = aln2[star_ind+1:].replace('-', '')

	return aln1, aln2

def rotate_left(aln, rot_len=1):
	return aln[rot_len:] + aln[:rot_len]

def rotate_right(aln, rot_len=1):
	return aln[-1*rot_len:] + aln[:-1*rot_len]

def star_move_left(aln, mov_len=1):
	star_ind = aln.index('*')
	part1 = aln[:star_ind]
	part2 = aln[star_ind+1:]
	new_part1 = part1[:-1*mov_len]
	new_part2 = part1[-1*mov_len:] + part2
	return new_part1 + '*' + new_part2

def star_move_right(aln, mov_len=1):
	star_ind = aln.index('*')
	part1 = aln[:star_ind]
	part2 = aln[star_ind+1:]
	new_part1 = part1 + part2[:mov_len]
	new_part2 = part2[mov_len:]
	return new_part1 + '*' + new_part2
