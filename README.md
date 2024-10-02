# GINClus: RNA structural motif clustering using graph isomorphism network
###### Authors: Created by Nabila Shahnaz Khan in collaboration with Md Mahfuzur Rahaman and Shaojie Zhang
GINClus source code is implemented using __Python 3.8.10__ and can be executed in __64-bit Linux__ machine. It is compatible with __Python 3.8.10-3.11.10__. For given RNA motif and motif candidate locations from PDB, GINClus generates the graph representation for each motif/motif candidate. Then, it generates motif subclusters based on their structural (base interaction and 3D structure) similarity. It can also generate side-by-side images of motifs for each subclusters (optional).  



## Install Instructions 

#### Install python3:
```
Debian/Ubuntu: apt install python3.8  
Fedora/CentOS: dnf install python3.8 
```

#### Install pip3: 
```
Debian/Ubuntu: apt install python3-pip  
Fedora/CentOS: dnf install python3-pip  
```

#### Install required Python libraries:  
The python libraries required to run GINClus are included in the [requirements.txt](requirements.txt) file. To install all required python libraries, please navigate to the GINClus home directory in the terminal and execute the following command.
```
pip3 install -r requirements.txt
```

#### Existing python packages:  
os, sys, shutil, math, random, subprocess, glob, time, argparse, logging, requests  
  
*** If any of the above mentioned package doesn't exist, then please install with command 'pip3 install package-name' ***


#### Create virtual environment (optional): 
For latest version of Ubuntu (>=23.0), all python packages have to be installed under a virtual environment. Use the follwing instructions to create a virtual envionment folder to run requirements.txt and GINClus code inside it:

```
mkdir GINClus_venv  
python3 -m venv GINClus_venv
source GINClus_venv/bin/activate  
```

#### Install PyMOL (optional - required to generate images):  
```
sudo apt-get install -y pymol
```

If running GINClus inside virtual environment then use the following command to add the path of pymol to the virtual environment path:

```
python3 -m venv GINClus_venv --system-site-packages
```

PyMOL can also be installed directly by downloading the OS-specific version from https://pymol.org/. However, the open-source PyMOL can be obtained by compiling the source code(Python 3.6+ is required). The steps to compile the PyMOL source code is described in the [README-PyMOL-build](README-PyMOL-build.md) file.


## Run Instructions
      
**_Run command:_** python3 run.py [-i1 'Train_motif_location_IL_PDB_input.csv'] [-i2 'Unknown_motif_location_IL_PDB_input.csv'] [-o 'output/'] [-e 0] [-d web] [-idt pdb] [-t True] [-idx 0] [-w 1] [-val 0.064] [-test 0.063] [-f True] [-c True] [-k 400] [-p False]  
**_Help command:_** python3 run.py -h  
**_Optional arguments:_** 
```
  -h, --help    show this help message and exit
  -i1 [I1]      Input file containing training motif locations. Default:'Train_motif_location_IL_input_PDB.csv'.
  -i2 [I2]      Input file containing motif candidate locations. Default:'Unknown_motif_location_IL_input_PDB.csv'.
  -o [O]        Path to the output files. Default: 'output/'.
  -e [E]        Number of extended residues beyond loop boundary to generate the loop.cif file. Default: 0.
  -d [D]        Use 'tool' to generate annotation from DSSR tool, else use 'web' to generate annotation from DSSR website. Default: 'web'.
  -idt [IDT]    Use 'fasta' if input motif index type is FASTA, else use 'pdb' if input motif index type is PDB. Default: 'pdb'.
  -t [T]        Trains the model if t = True, else uses the previously trained model weight. To set the parameter to False just use '-t'. Default: True.
  -idx [IDX]    Divides data into train, validation and test. To divide randomly, set to '0'. To divide according to the paper for internal loops, set to '1'. To divide according to the paper for
                hairpin loops, set to '2'. To define manually using the file 'Train_Validate_Test_data_list.csv' in data folder, set to '3'. Default: 0.
  -w [W]        Use '1' to save the new model weight, otherwise, use '0'. Default: '1'.
  -val [VAL]    Set the percentage of validation data. Default: '0.064'.
  -test [TEST]  Set the percentage of test data. Default: '0.063'.
  -f [F]        Generates features for unknown motifs if True, else uses the previously generated features. To set the parameter to False just use '-f'. Default: True.
  -c [C]        Generates cluster output if True, else uses the previously generated clustering output. To set the parameter to False just use '-c'. Default: True.
  -k [K]        Define the number of clusters (value of K) to be generated. Default: 400.
  -p [P]        If True, generates PyMOL images for output clusteres. Default: False.

```

**_Input:_** GINClus takes the locations of known RNA motifs and RNA motif candidates (loop regions) as input from two separate input files inside the [data](data/) folder. Theses locations are expected to be in PDB index, but it can be changed into FASTA index by setting the "-idt" parameter to "fasta".
1. __Train_motif_location_input:__ contains the locations of RNA motifs used for training. These locations can be collected from PDB files. Each line in the input file starts with the family name, followed by RNA motif locations. The motif locations are provided using the format 'PDBID_CHAIN:locations'. Example motif location: '4LCK_B:66-71', '1U9S_A:61-64_86-87', '4RGE_C:10-13_24-25_40-43'. Example input files: 'Train_motif_location_IL_input_PDB.csv', 'Train_motif_location_HL_input_PDB.csv'.
2. __Unknown_motif_location_input:__ contains the locations of RNA motif candidates. Uses the similar format as the file Train_motif_location_input files. Example input files: 'Unknown_motif_location_IL_input_PDB.csv', 'Unknown_motif_location_HL_input_PDB.csv'.


**_Optional Input File:_** The optional input file example can be found inside the [data](data/) folder.
__Train_Validate_Test_data_list.csv:__ To manually define the locations of motifs that should be used for training, validation and testing, use this input file. The first section of the file contains the locations of the training motifs, the second section contains the locations of the validation motifs and the third section contains the locations of the testing motifs.


**_Output:_** Generates the following output files inside the user defined output folder, default:'output'.
1. __Motif_candidate_features.tsv:__ contains the feature set generated for each motif. These features are used to cluster the RNA motifs.
2. __Cluster_output.csv:__ contains the clustering output of RNA motifs generated by K-means clustering algorithm.
3. __Subcluster_output.csv:__ contains the subclustering output of RNA motifs generated by Hierachical Agglomerative clustering algorithm.
4. __subcluster_images folder:__ contans the images generated for each subcluster. The images will be generated if PyMOL is installed and "-p" parameter used is while running GINClus.


**_Example run commands for sample input:_**
Example sample inputs are provided inside [data](data/) folder. More example sample input files can be found inside folder [sample_input](data/sample_input/).
1. __For internal loops:__ 
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB.csv' -o 'output/' -idt pdb -d web -e 0 -idx 1 -w 1 -val 0.064 -test 0.063 -k 400
```
2. __For hairpin loops:__ 
```
python3 run.py -i1 'Train_motif_location_HL_input_PDB.csv' -i2 'Unknown_motif_location_HL_input_PDB.csv' -o 'output/' -idt pdb -d web -e 0 -idx 2 -w 1 -val 0.064 -test 0.063 -k 400
```
3. __For fasta index:__ 
```
python3 run.py -i1 'sample_input/Train_motif_location_HL_input_FASTA.csv' -i2 'sample_input/Unknown_motif_location_HL_input_FASTA.csv' -o 'output/' -idt fasta -d web -e 0 -idx 0 -w 1 -val 0.064 -test 0.063 -k 400
```
4. __For image generation:__ 
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB.csv' -idx 1 -p
```
       
### Important Notes
*** The PDB files 6EK0 and 4P95 have recently become obsolete and the motifs previously collected from these RNA chains have been removed from the motif location input files.  
*** Input files containing motif locations are provided inside folder [data](data/) and [sample_input](data/sample_input/).  
*** The GINClus clustering results for [internal loops](output/Subcluster_output_IL.xlsx) and [hairpin loops](output/Subcluster_output_HL.xlsx) discussed in the paper are provided inside folder [output](output/).  
*** Generating images for subclusters is optional and it takes comparatively longer to generate all the images.  
*** Needs to install/download PyMOL to generate the subcluster images using GINClus.  
*** GINClus can cluster multiloops, however we didn't have enough training data to generate authentic multiloop clusters. We provided sample input files 'Train_motif_location_ML_input_PDB.csv' and 'Unknown_motif_location_ML_input_PDB.csv' to show that GINClus can handle multi-loops.  
*** GINClus can handle internal loops, hairpin loops and multiloops at the same time. We provided sample input files 'Train_motif_location_mixed_input_PDB.csv' and 'Unknown_motif_location_mixed_input_PDB.csv' to show that GINClus can handle them at the same time.   


### Terms  
Where appropriate, please cite the following GINClus paper:  
Nabila et al. "GINClus: RNA Structural Motif Clustering Using Graph Isomorphism Network." 


### Contact
For any questions, please contact nabila.shahnaz.khan@ucf.edu
