# GINClus: RNA structural motif clustering using graph isomorphism network
###### Authors: Created by Nabila Shahnaz Khan in collaboration with Md Mahfuzur Rahaman and Shaojie Zhang
GINClus source code is implemented using __Python 3.8.10__ and can be executed in __64-bit Linux__ machine. For given RNA motif locations (PDB or FASTA), GINClus first collects the RNA motifs and then generates the graph representation for each motif. Finally, it generates motif subclusters based on their structural (base interaction and 3D structure) similarity. It can also generate side-by-side images of motifs for each subclusters (optional).  



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


#### Install PyMOL (optional - required to generate images):  
PyMOL can be installed directly by downloading the OS-specific version from https://pymol.org/. However, the open-source PyMOL can be obtained by compiling the source code(Python 3.6+ is required). The steps to compile the PyMOL source code is described in the [README-PyMOL-build](README-PyMOL-build.md) file.


## Run Instructions
    
  
**_Run command:_** python3 run.py [-i1 'Train_motif_location_input.csv'] [-i2 'Unknown_motif_location_input.csv'] [-o 'output/'] [-e 0] [-t False] [-idx 0] [-w 1] [-val 0.064] [-test 0.063] [-f False] [-c False] [-k 400] [-tf 0] [-uf 0] [-p False]  
**_Help command:_** python3 run.py -h  
**_Optional arguments:_** 
```
  -h, --help  	show this help message and exit 
  -i1 [I1]    	Input file containing motif locations for training. Default:'Train_motif_location_input.csv'.  
  -i2 [I2]    	Input file containing motif locations for unknown families. Default:'Unknown_motif_location_input.csv'.
  -o [O]      	Path to the output files. Default: 'output/'.   
  -e [E]      	Number of extended residues beyond loop boundary to generate the loop.cif file. Default: 0.  
  -t [T]      	Trains the model if t = True, else uses the previously trained model weight. Default: False. 
  -idx [IDX]  	Divides data into train, validation and test. To divide according to the paper, set to '0'. To divide randomly, set to '1'. To define manually using the file 'Train_Validate_Test_data_list.csv' in data folder, set to '2'. Default: 0.
  -w [W]      	Use '1' to save the new model weight, otherwise, use '0'. Default: '1'.  
  -val [VAL]  	Set the percentage of validation data. Default: '0.064'.
  -test [TEST]	Set the percentage of test data. Default: '0.063'.
  -f [F]	Generates features for unknown motifs if f = True, else uses the previously generated features. Default: False.
  -c [C]	Generates cluster output if t = True, else uses the previously generated clustering output. Default: False.
  -k [K]	Define the number of clusters (value of K) to be generated. Default: 400.
  -tf [TF]	If tf = 1, takes RNA motif family information for train data from 'Train_family_info.csv' in the data folder. Otherwise uses the existing family information. Default: 0.
  -uf [UF]	If uf = 1, takes RNA motif family information for unknown motif data from 'Unknown_motif_family_info.csv' in the data folder. If uf = 2, only uses 'IL' as family name for unknown motifs. Otherwise, uses the existing family information for unknown motifs. Default: 0.
  -p [P]	If True, generates PyMOL images for output clusteres. Default: False.
	  
```

**_Input:_** GINClus takes RNA motif locations as input for both training motifs and unknowns motifs from the following two separate files inside the [data](data/) folder.
1. __Train_motif_location_input.csv:__ contains the locations of RNA motifs used for training. These locations can either be collected from FASTA files or PDB files. Each line in the input file starts with the family name, followed by RNA motif locations. The motif locations are provided using the format 'PDBID_CHAIN:pos1_pos2_pos3_pos4'. Example motif location: '1U9S A:61-64 86-87'.
2. __Unknown_motif_location_input.csv:__ contains the locations of unknown RNA motifs (PDB or FASTA) used for training. Uses the similar format as the file 'Train_motif_location_input.csv'.


**_Optional Input Files:_** All the optional input file examples can be found inside the [data](data/) folder.
1. __Train_family_info.csv:__ this input file contains RNA family names of training RNA motifs. Column 1 contains the family name and column 2 contains the family name prefix.
2. __Unknown_motif_family_info.csv:__ this input file contains RNA family names of internal loop motif instances (in case some of the internal loops' family names are known). Column 1 contains the family name and column 2 contains the family name prefix. Please use family name 'Unknown internal loop' with prefix 'IL' for all the unknown internal loop motif instances. In case all the internal loops' family information is unknown, use the parameter uf = 2.
3. __Train_Validate_Test_data_list.csv:__ To manually define the locations of motifs that should be used for training, validation and testing, use this input file. The first section of the file contains the locations of the training motifs, the second section contains the locations of the validation motifs and the third section contains the locations of the testing motifs.


**_Output:_** Generates the following output files inside the user defined output folder, default:'output'.
1. __Motif_Features_IL.tsv:__ contains the feature set generated for each motif. These features are used to cluster the RNA motifs.
2. __Cluster_output.csv:__ contains the clustering output of RNA motifs generated by K-means clustering algorithm.
3. __Subcluster_output.csv:__ contains the subclustering output of RNA motifs generated by Hierachical Agglomerative clustering algorithm.
4. __subcluster_images folder:__ contans the images generated for each subcluster.


       
### Important Notes
*** The PDB files 6EK0 and 4P95 have recently become obsolette and the motifs previously collected from these RNA chains have been removed from the motif location input file 'Unknown_motif_location_input.csv'.  
*** Input files containing motif locations are provided inside folder [data]('data/') and the [GINClus clustering results](output/Subcluster_output.xlsx) discussed in the paper are provided inside folder [output](output/).  
*** All the optional input files (Train_family_info.csv, Unknown_motif_family_info.csv, Train_Validate_Test_data_list.csv) need to be inside the data folder prior to the run.  
*** Generating images for subclusters is optional and it takes comparatively longer to generate all the images.  
*** Needs to download the open-source PyMOL to generate the subcluster images using GINClus.  


### Terms  
Where appropriate, please cite the following GINClus paper:  
Nabila et al. "GINClus: RNA Structural Motif Clustering Using Graph Isomorphism Network." 


### Contact
For any questions, please contact nabila.shahnaz.khan@ucf.edu
