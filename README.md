# GINClus: RNA Structural Motif Clustering Using Graph Isomorphism Network


## Install Instructions 
GINClus source code is implemented using Python 3.8.10 and can be executed in 64-bit Linux machine. The implementation can be divided into three major parts. (i) For the given motif locations, generating the graph representations for those motifs (ii) Generating motif subclusters based on their structural (base interaction and 3D structure) similarity (iii) For each subcluster, generating 3D diagrams of motifs side by side (This step is optional and can take comparatively longer to run)


```
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
It is required to install several python libraries to run RNA-NRD pipeline. These libraries are included in the [requirements.txt](requirements.txt) file. To install all required python libraries, please navigate to the RNA-NRD home directory in the terminal and execute the following command.

```
pip3 install -r requirements.txt
``` 

#### Python packages that should already exist:  
os, sys, shutil, math, random, subprocess, glob, time, argparse, logging, requests  
  
*** If any of the above mentioned package doesn't exist, then please install with command 'pip3 install package-name' ***

## Run Instructions
    
  
**_Run command:_** python3 Run.py [-i 'Data.txt'] [-o1 'RNA-NRD_Dataset'] [-o2 'RNA-NRD_without_Organism_Division_Dataset'] [-t 80] [-r 4] [-a 80] [-org True]  
**_Help command:_** python3 Run.py -h  
**_Optional arguments:_** 
```
  -h, --help  show this help message and exit  
  -i [I]      Input file name containing RNA chains. Default: 'Data.txt'.  
  -o1 [O1]    Output file name with organism division. Default: 'RNA-NRD_Dataset'.  
  -o2 [O2]    Output file name without organism division. Default: 'RNA-NRD_without_Organism_Division_Dataset'.  
  -t [T]      Provide sequence identity threshold. Default: 80.  
  -r [R]      Provide RMSD threshold. Default: 4.  
  -a [A]      Provide structural alignment ratio threshold. Default: 80.  
  -org [ORG]  Generate RNA-NRD-without-Organism-Division dataset. Default: True. 


  -h, --help  show this help message and exit  
  -i [I]      Input file name containing RNA chains. Default: 'Data.txt'.  
  -o1 [O1]    Output file name with organism division. Default: 'RNA-NRD_Dataset'.  
  -o2 [O2]    Output file name without organism division. Default: 'RNA-NRD_without_Organism_Division_Dataset'.  
  -t [T]      Provide sequence identity threshold. Default: 80.  
  -r [R]      Provide RMSD threshold. Default: 4.  
  -a [A]      Provide structural alignment ratio threshold. Default: 80.  
  -org [ORG]  Generate RNA-NRD-without-Organism-Division dataset. Default: True. 
  
```


**_Output:_** Generates final nonredundant detaset output file inside [Output](/Output/) folder (Default: 'Output/RNA-NRD_Dataset')  


### Collect RNA chains from PDB (optional)
RNA chains have been collected from PDB on February 23, 2022 and provided inside the [Data](/Data/) folder under the name [Merged_data.txt](/Data/Merged_data.txt). But as [Merged_data.txt](/Data/Merged_data.txt) contains a huge number of RNA chains, it will take long time to run the pipeline for this input. Smaller samples of input data ([Data.txt](/Data/Data.txt), [Data_2.txt](/Data/Data_2.txt), [Data_3.txt](/Data/Data_3.txt)) has been provide inside the [Data](/Data/) folder which contain small amount of RNA chains and can be run within few minutes. To collect RNA chains from PDB, please complete the following steps:

**_Steps:_**
1. Download RNA chains from [PDB](https://www.rcsb.org/) in .csv format. Go to [Homepage](https://www.rcsb.org/) -> [“Nucleic Acid Containing Structures”](https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22entity_poly.rcsb_entity_polymer_type%22%2C%22negation%22%3Afalse%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22DNA%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22entity_poly.rcsb_entity_polymer_type%22%2C%22negation%22%3Afalse%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22RNA%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22entity_poly.rcsb_entity_polymer_type%22%2C%22negation%22%3Afalse%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22NA-hybrid%22%7D%7D%5D%2C%22logical_operator%22%3A%22or%22%7D%5D%2C%22logical_operator%22%3A%22and%22%2C%22label%22%3A%22text%22%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22results_content_type%22%3A%5B%22experimental%22%5D%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%2C%22scoring_strategy%22%3A%22combined%22%7D%2C%22request_info%22%3A%7B%22query_id%22%3A%22b58e075a4d9a0a80000fc64c88aaab46%22%7D%7D) -> “Tabular Report” -> “Create CustomReport” -> select attributes -> “Run Report”.
2. Select these 12 attributes: Experimental Method, Release Date, PDB ID, Number of Distinct RNA Entities, Resolution (Å), Sequence, Entity Polymer Type, Polymer Entity Sequence Length, Source Organism, Macromolecule Name, Chain ID (Asym ID), Entry Id (Polymer Entity Identifier).
3. Convert .csv files to .txt file format and merge them. See the example code below and change the file names accordingly:
		
		****************************************************************************************
		csvformat -T rcsb_pdb_custom_report_0001-2500.csv > rcsb_pdb_custom_report_0001-2500.txt
		sed -i 's/\ /_/g' rcsb_pdb_custom_report_0001-2500.txt

		csvformat -T rcsb_pdb_custom_report_2501-5000.csv > rcsb_pdb_custom_report_2501-5000.txt
		sed -i 's/\ /_/g' rcsb_pdb_custom_report_2501-5000.txt

		csvformat -T rcsb_pdb_custom_report_5001-5329.csv > rcsb_pdb_custom_report_5001-5329.txt
		sed -i 's/\ /_/g' rcsb_pdb_custom_report_5001-5329.txt

		cat *.txt > Data.txt
		****************************************************************************************
		

### Collect RNA chain family information from Rfam website (optional)  
RNA chain family information has been collected from Rfam website on January 23, 2023 and provided inside the folder [Rfam_family](/Rfam_family/). In order to collect most recent Rfam family information, please complete the following steps:

**_Requirement:_** Install chromedrive and update the chromedrive path in the code [Rfam_parser.py](/Rfam_family/Rfam_parser.py) on line 26   
**_Instructions:_** Run the code inside the folder [Rfam_parser.py](/Rfam_family/Rfam_parser.py). If chromedriver not found, then update the 'DRIVER_PATH' in the code.     
**_Run command:_** python3 Rfam_parser.py  
**_Output:_** generates file [PDB_family_name.txt](/Rfam_family/PDB_family_name.txt) containing Rfam family information insdie the folder [Rfam_family](/Rfam_family/).   
           
### Important Notes
*** Please make sure the file [All_Organism_list](/src/Organism_list/All_Organism_list) is inside the folder [Organism_list](/src/Organism_list/). It contains list of all organisms currently present in PDB.

### Terms  
Where appropriate, please cite the following RNA-NRD paper:  
Nabila et al. "RNA-NRD: a Non-redundant RNA Structural Dataset for Benchmarking and Functional Analys." 

### Contact
For any questions, please contact nabilakhan@knights.ucf.edu
