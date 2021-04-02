# CarSNN
Implementation of CarSNN:  An Efficient Spiking Neural Network for Event-Based Autonomous Cars on the Loihi Neuromorphic Chip

CarSNN is a novel SNN model for the “cars vs. background” classification of event-based streams implemented on neuromorphic hardware. 
In these python script we use Spatio Temporal BackPropagation (STBP) learning rule to train the network.
We use also three different attention windows on the input image in order to speedup the train and test process and to achieve good results.
We adopt an accumulation strategy and we give to the network an input image every 1 ms and we predict the class. After 10 sample (so 10 ms) we choose the true classification according to the most predicted. 

It achieves on the N-cars dataset the following accuracy after 10 samples
* CarSNN (128×128attention window) : 0.86
* CarSNN (100×100attention window) : 0.86
* CarSNN (50×50attention window) : 0.79
