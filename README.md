# CarSNN
Implementation of CarSNN:  An Efficient Spiking Neural Network for Event-Based Autonomous Cars on the Loihi Neuromorphic Chip

CarSNN is a novel SNN model for the “cars vs. background” classification of event-based streams implemented on neuromorphic hardware. 
In these python script we use Spatio Temporal BackPropagation (STBP at link https://arxiv.org/pdf/1706.02609.pdf) learning rule to train the network.
We use also three different attention windows on the input image in order to speedup the train and test process and to achieve good results.
We adopt an accumulation strategy and we give to the network an input image every 1 ms and we predict the class. After 10 sample (so 10 ms) we choose the true classification according to the most predicted. 

It achieves on the N-cars dataset the following accuracy after 10 samples
* CarSNN (128×128attention window) : 0.86
* CarSNN (100×100attention window) : 0.86
* CarSNN (50×50attention window) : 0.79

## Instruction to run the code

The following parts describe how to use this code.
To run this code you must download the dataset from the link https://www.prophesee.ai/2018/03/13/dataset-n-cars/ and then you must change the format of the images with the provided matlab code. Then you can train the network and visualize the test results.

### Matlab script to study the N-cars dataset and dataset format

With the script Occurrences_and_translation.m you can derive the event occurences for every pixel both negative and positive.

This code is also usefull to cahnge the format of the inputs. The python code would at input .DAT files that describe the spike trace of the event streams.
These files have 4 column with the following format:

"timestamp of event in usec" "x coordinate" "y coordinate" "polarity of event (-1 or 1)"

If you desire you can write your own dataset and give your images with the above format without change nothing

### Network format

The networks are descripted into .txt files in the net directory. You can find all the used network with and without ceil mode for the pooling layers. You can set this mode by modify the rows 250, 251 and 276 of the file spiking_model_LIF.py . 
The network files are only files with as many rows as the layers used every row identify a layer we give some example to make it easy to understand:
* Convolution layers row format: 
  "input channels" "output channels" "stride" "padding (1 for 'same', 0 for 'valid')" "kernel size" "groups". 
  Example 1 8 2 1 3 1 describe a convolution layer with 1 input channel, 2 output channel, stride of size 2x2, kernel size of 3x3, same padding and 1 for groups.
* Dense layers row format: 
  "size of input" "size of output".
  Example: 1024 2048 describe a fully connected layer with for the input 1024 neurons and for the output 2048.
* Average pooling layers row format: 
  "stride (number<10)". 
  Example: 2 describe an average pooling layer with stride size equal to kernel size 2x2.
* Dropout layers row format: 
  "dropout percentage (number>=10)".
  Example: 15 describe a dropout layer with dropout percentage equal to 15%.

### Command to run the python code and train and test the network

To run the network we define some arguments to give at the command line:
* --filenet : is the path of the network we would use. Example: --filenet ./net/net_1_4a32c3z2a32c3z2a_100_100.txt .
* --fileresult : is the path to save a txt file with some initialization information and the accuracy achieved at every epoch. Default: './result.txt' .
* --sample_time : for many ms we would accumulate the events before give them to the network. Default: 1 . 
* --sample_length : how many accumulated sample we use to find the class of the stream. Default: 10 . 
* --batch_size : the batch size used both in train and test. Default: 40 .
* --lr : learning rate. Default: 1e-3 .
* --lr_decay_epoch : after many epoch we would ciclically modify the lr by --lr_decay_value factor. Default: 20 .
* --lr_decay_value : how to modify the learning rate. At every lr_decay_epoch we multiply the old learning rate value with this factor that can be greater or less than zero. Default: 0.5 . 
* --threshold : the neuron threshold after that we have an output spike. Default: 0.4 .
* --n_decay : decay value for the membrane potential of every neurons used (we suggest a number less or equal to 0.2). Default: 0.2 .
* --att_window : the attention window size and shift. Example --att_window 50 60 20 10 describe and attention window of x size of 50, y size of 60, x shift of 20 and y shift of 10
* --weight_decay : it is the weight regularization described for the Adam optimizer at the link https://arxiv.org/abs/1711.05101 . Default: 0 .
To change the epoch to run the train you have to change the parameter num_epoch in spiking_model_LIF.py script at row 13

Example of command to run the code:
```
CUDA_VISIBLE_DEVICES=0 python STBP_dvs_n_car.py --filenet ./net/net_1_4a32c3z2a32c3z2a_100_100.txt --fileresult res_prova_input_100_100_st_1_sl_10_bs_40_15tw_2_ch_trained.txt --batch_size 40 --channel 2 --lr 1e-3 --lr_decay_epoch 20 --lr_decay_value 0.5 --threshold 0.4 --att_window 100 100 0 0 --sample_length 10 --sample_time 1
```
