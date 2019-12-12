# GPASAN
We setup our experiment on an Nvidia Titan GPU and 256G memory on CentOs. 

Experiment Environment
-------
* python 3.6.5
* tensorflow-gpu 1.12
* numpy

Project Struct
------
### data
* taobao_buy_8_time_4.txt  ------ the dataset of taobao buy, Each line is in format: UserID \t ItemID \t CategoryId \t Time. we don't use the category information, It's just for catering to some other datasets' format.
* #### adj_matrix        
### program
* main.py                   ------ this is the program entry
* model_time_two_channel_gcn.py             ------ the ReGNN model
* utils.py                  ------ some helper functions we used

Recommended Setup
------
You can run the main.py directly for easily running the program. 
If you run the code on linux, just running the following command:<br>
<br>
      `python main.py`
