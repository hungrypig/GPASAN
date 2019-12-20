# GPASAN
We setup our experiment on an Nvidia Titan GPU and 256G memory on CentOs. 

Experiment Environment
-------
* python 3.6.5
* tensorflow-gpu 1.12
* numpy

Project Struct
------
### GPASAN/data
* taobao_buy_8_time_4.zip  ------ the dataset of taobao buy, Each line is in format: UserID \t ItemID \t CategoryId \t Time. we don't use the category information, It's just for catering to some other datasets' format. Unzip the taobao_buy_8_time_4.zip in this directory before running the program.

* #### adj_matrix/taobao_buy_8 
------
* s_norm_adj_mat.npz  ------ the preprocessed adjacent matrix of the item-item graph
* taobao_buy_8.txt  ------ a temporary file

### GPASAN/
* main.py                   ------ this is the program entry
* model_time_two_channel_gcn.py             ------ the major PGASAN model
* modules_time.py                  ------ the periodic-attenuated self-attention network module
* sampler_time_repeat.py                  ------ The file used to create the training data
* util_periodic_two_channel.py                  ------ some helper and test functions we used

Recommended Setup
------
You can run the main.py directly for easily running the program. 
If you run the code on linux, just running the following command:<br>
<br>
      `python main.py`
