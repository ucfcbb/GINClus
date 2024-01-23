import sys
import os

sys.path.append('../../')
from config import *
sys.path.append(scripts_dir)
from utils import *
from image_helper import *

try:
    import pymol
    from pymol import stored
except Exception as e:
    try:
        sys.path.append(pymol_py3_dir)
        import pymol
        from pymol import stored
    except Exception as e:
        pass
        # logger.error('PyMOL not found.')


def load_pdb_res_map(chain):
    """load sequence index->pdb index"""
    """{ref_index: (chain_id, pdb_index)}"""
    ret = {}
    # map_dir = '../nrPDBs_Old' # the directory for the mapping data
    mapping_file_name = chain + '.rmsx.nch'
    # This was done to make the code compatible with any case-insensitive OS
    if chain != chain.upper():
        mapping_file_name = chain + '_.rmsx.nch'
    fp = open(os.path.join(pdb_fasta_mapping_dir, mapping_file_name))
    lines = fp.readlines()
    fp.close()
    for line in lines:
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

def reset_pymol():
    pymol.cmd.sync()
    # pymol.commanding.sync()
    pymol.cmd.deselect()
    pymol.cmd.delete('all')
    pymol.cmd.reinitialize()
    pymol.cmd.bg_color('white')

def get_index_list(loop):
    ind_list = []
    segments = loop.strip().split(':')[1].strip().split('_')
    for segment in segments:
        a, b = segment.strip().split('-')
        ind_list.append((int(a), int(b)))
    # print(ind_list)
    return ind_list

def get_seqnums_from_indices(indices):
    seqnums = []
    # print(indices)
    for chain, index, icode in indices:
        seqnums.append(index + icode)

    return seqnums

def get_loop_boundary_pdb_index(loop_boundary_fasta):
    loop_boundary_pdb = {}
    for r in loop_boundary_fasta:
        loop_boundary_pdb[r] = []
        pdb_chain = r.strip().split(':')[0]
        pdb_res_map = load_pdb_res_map(pdb_chain)
        for (a, b) in loop_boundary_fasta[r]:
            for ind in range(a, b+1):
                if ind in pdb_res_map:
                    loop_boundary_pdb[r].append(pdb_res_map[ind])

    return loop_boundary_pdb

def generate_pymol_images(scl_wise_loop_dict, pdb_ind_list_dict, image_output_dir, partial_pdbx_dir):
    pymol.finish_launching(['pymol', '-cqQ'])
    # pymol.finish_launching()    # show window, can be used to set view

    for scl_id in scl_wise_loop_dict:
        create_directory(os.path.join(image_output_dir, 'subcluster_' + str(scl_id)))

        reset_pymol()
        reference_load_name = ''
        for i, (loop, _) in enumerate(scl_wise_loop_dict[scl_id]):
            loop_PDB = convert_a_loop_from_FASTA_to_PDB(loop)
            
            pdb_load_name = loop_PDB.replace(':', '_')
            display_load_name = 'display_' + pdb_load_name
            display_color = 'red'
            if i == 0:
                reference_load_name = pdb_load_name

            pdb_disp_ind_list = get_seqnums_from_indices(pdb_ind_list_dict[loop])
            pymol.cmd.load(os.path.join(partial_pdbx_dir, loop.replace(':', '_')+'.cif'), pdb_load_name)
            
            pymol.cmd.select(display_load_name, pdb_load_name + ' and (%s)' % ' or '.join(list(map(lambda x: 'resi '+x, pdb_disp_ind_list))))

            pymol.cmd.sync()
            pymol.cmd.color('gray', pdb_load_name)
            pymol.cmd.color(display_color, display_load_name)
            # pymol.cmd.hide('everything', pdb_load_name)
            pymol.cmd.deselect()

        pymol.cmd.alignto(reference_load_name, 'align')
        # pymol.cmd.alignto(reference_load_name, 'super')
        pymol.cmd.zoom()
        pymol.cmd.sync()
        pymol.cmd.hide()

        image_file_list = []

        for i, (loop, filename) in enumerate(scl_wise_loop_dict[scl_id]):
            loop_PDB = convert_a_loop_from_FASTA_to_PDB(loop)
            pdb_load_name = loop_PDB.replace(':', '_')
        #     display_load_name = 'display_' + pdb_load_name

            image_fname = os.path.join(image_output_dir, 'subcluster_' + str(scl_id), filename.strip().split('_')[0] + '_' + pdb_load_name + '.png')
            image_file_list.append((loop_PDB, image_fname, 1))

            pymol.cmd.hide()
            pymol.cmd.show('cartoon', pdb_load_name)
            pymol.cmd.sync()
            pymol.cmd.zoom()
            pymol.cmd.png(image_fname, 1200, 1200, dpi=300, ray=1, quiet=1)
            pymol.cmd.sync()
            # sys.exit()

        # if int(scl_id) > 60:
        #     break

        create_collage(image_file_list, os.path.join(image_output_dir, 'subcluster_' + str(scl_id) + '.png'), True)
        # create_collage(image_file_list, pdb_organism_details, os.path.join(subfamily_dir, str(cluster_id) + '_' + suffix + '.png'), show_pdb_info, is_graph_image, show_image_caption)

        pymol.cmd.hide()
        pymol.cmd.sync()
        
        # superimposition image generate
        # for i, (loop, filename) in enumerate(scl_wise_loop_dict[scl_id]):
        #     loop_PDB = convert_a_loop_from_FASTA_to_PDB(loop)
        #     pdb_load_name = loop_PDB.replace(':', '_')
        #     pymol.cmd.show('cartoon', pdb_load_name)
        # pymol.cmd.zoom(reference_load_name)
        # image_fname = os.path.join(image_output_dir, 'subcluster_' + str(scl_id) + '_superimposed.png')
        # pymol.cmd.png(image_fname, 1200, 1200, dpi=300, ray=1, quiet=1)
        # pymol.cmd.sync()


    pymol.cmd.quit()
    for scl_id in scl_wise_loop_dict:    
        # wait_for_certain_files_to_be_generated(['subcluster_' + str(scl_id) + '.png', 'subcluster_' + str(scl_id) + '_superimposed.png'])
        wait_for_certain_files_to_be_generated(['subcluster_' + str(scl_id) + '.png'])
        # print('deleting contents')
        remove_all_from_dir(os.path.join(image_output_dir, 'subcluster_' + str(scl_id)))
        delete_directory(os.path.join(image_output_dir, 'subcluster_' + str(scl_id)))

def generate_motif_images(input_fname, output_path, partial_pdbx_dir):
    fp = open(input_fname, 'r')
    lines = fp.readlines()
    fp.close()
    scl_wise_loop_dict = {}
    fasta_ind_list_dict = {}
    for line in lines[1:]:
        pieces = line.strip().split('\t')
        # subpieces = pieces[0].strip().split('_')
        # loop = '_'.join(subpieces[1:3]) + ':' + '_'.join(subpieces[3:])
        loop = pieces[0]
        scl_id = pieces[2].strip()
        if scl_id not in scl_wise_loop_dict:
            scl_wise_loop_dict[scl_id] = []
        scl_wise_loop_dict[scl_id].append((loop, pieces[3].strip() + '_' + pieces[0].strip()))
        if loop not in fasta_ind_list_dict:
            fasta_ind_list_dict[loop] = get_index_list(loop)

    pdb_ind_list_dict = get_loop_boundary_pdb_index(fasta_ind_list_dict)

    generate_pymol_images(scl_wise_loop_dict, pdb_ind_list_dict, os.path.join(output_path, 'subcluster_images'), partial_pdbx_dir)