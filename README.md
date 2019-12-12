# GPASAN
We setup our experiment on an Nvidia Titan GPU and 256G memory on CentOs. 

Experiment Environment
-------
* python 3.6.5
* tensorflow-gpu 1.12
* numpy

Project Struct
------
### datasets
* YoochooseSubDataset file  ------ this file contains a relatively small subdataset of Yoochoose to run test the code and experiment results
* data_preprocess.py        ------ this script is used to preprocess the origin data and split train dataset and test dataset
### program
* main.py                   ------ this is the program entry
* ReGNN.py                  ------ the ReGNN model
* utils.py                  ------ some helper functions we used

Recommended Setup
------
You can run the main.py directly for easily running the program. 
If you run the code on linux, just running the following command:<br>
<br>
      `python main.py`
