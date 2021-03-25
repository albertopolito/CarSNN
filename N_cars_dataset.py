import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

"""
In this file is presented the management of the N-Cars dataset.

The dataset is presented by a tree of directories:

    -train : train part of dataset
        + cars : contains all samples that represent cars 
        + background : contains all samples that represent background

    -test : train part of dataset
        + cars : contains all samples that represent cars
        + background : contains all samples that represent background

With the classes in this file we can read samples and adapt them for torch tensor in order to send them to the network.

"""


class event():
	'''
	This class provides a way to read spike event.

	Members:
		* ``x`` : `x` index of spike event.
		* ``y`` : `y` index of spike event (not used if the spatial dimension is 1).
		* ``p`` : `polarity` or `channel` index of spike event.
		* ``t`` : `timestamp` of spike event. Time is assumend to be in ms.

	Usage:

	>>> TD = spikeFileIO.event(xEvent, yEvent, pEvent, tEvent)
	'''
	def __init__(self, xEvent, yEvent, pEvent, tEvent):
		if yEvent is None:
			self.dim = 1
		else:
			self.dim = 2

		self.x = xEvent if type(xEvent) is np.ndarray else np.asarray(xEvent) # x spatial dimension
		self.y = yEvent if type(yEvent) is np.ndarray else np.asarray(yEvent) # y spatial dimension
		self.p = pEvent if type(pEvent) is np.ndarray else np.asarray(pEvent) # spike polarity
		self.t = tEvent if type(tEvent) is np.ndarray else np.asarray(tEvent) # time stamp in ms

		if not issubclass(self.x.dtype.type, np.integer): self.x = self.x.astype('int')
		if not issubclass(self.p.dtype.type, np.integer): self.p = self.p.astype('int')
		
		if self.dim == 2:	
			if not issubclass(self.y.dtype.type, np.integer): self.y = self.y.astype('int')
		# the dataset files describe -1 as negative event and 1 as positive event
		self.p -= self.p.min()  # self.p has values from 0 to 2 (3 event types, but only the first and the thrird are used)
		self.p = (self.p/2)     # self.p has values from 0 (negative event) to 1 (positive event) 
		self.p = self.p.astype('int')

	
	def toSpikeTensor(self, emptyTensor, samplingTime=1, randomShift=True, shift_x=0, shift_y=0):	# Sampling time in ms
		'''
		Returns a numpy tensor that contains the spike events sampled in bins of `samplingTime`.
		The tensor is of dimension (channels, height, width, time) or``CHWT``.

		Arguments:
			* ``emptyTensor`` (``numpy or torch tensor``): an empty tensor to hold spike data 
			* ``samplingTime``: the width of time bin to use.
			* ``randomShift``: flag to shift the sample in time or not. Default: True.
			* ``shift_x``: value of pixel to shift the attention window on x coordinate. Default: 0.
			* ``shift_y``: value of pixel to shift the attention window on y coordinate. Default: 0.
		Usage:

		>>> spike = TD.toSpikeTensor( torch.zeros((2, 240, 180, 5000)) )
		'''
		
		if randomShift is True:
			tSt = np.random.uniform(
				max(
					int(self.t.min() / samplingTime), 
					int(self.t.max() / samplingTime) - emptyTensor.shape[3],
					emptyTensor.shape[3] - int(self.t.max() / samplingTime),
					1,
				)
			)
		else:
			tSt = 0
		
		xEvent = np.round(self.x).astype(int)
		pEvent = np.round(self.p).astype(int)
		tEvent = np.round(self.t/samplingTime).astype(int) - tSt

		
		if self.dim == 1:
			# with the following code we involve the accumulation of events and attention window
			validInd = np.argwhere((xEvent < emptyTensor.shape[2]+shift_x) &
								   (tEvent < emptyTensor.shape[3]) &
								   (xEvent >= 0+shift_x) &
								   (tEvent >= 0))
			emptyTensor[pEvent[validInd],
						0, 
				  		xEvent[validInd]-shift_x,
				  		tEvent[validInd]] = 1
		elif self.dim == 2:
			# with the following code we involve the accumulation of events and attention window
			yEvent = np.round(self.y).astype(int)
			validInd = np.argwhere((xEvent < (emptyTensor.shape[2]+shift_x)) &
								   (yEvent < (emptyTensor.shape[1]+shift_y)) & 								   
								   (tEvent < emptyTensor.shape[3]) &
								   (xEvent >= 0+shift_x) &
								   (yEvent >= 0+shift_y) & 								   
								   (tEvent >= 0))

			emptyTensor[pEvent[validInd], 
					yEvent[validInd]-shift_y,
					xEvent[validInd]-shift_x,
			 		tEvent[validInd]] = 1
		return emptyTensor


def read2Dspikes(filename):
	'''
	Reads two dimensional binary spike file and returns corresponding events.
	
	The dataset files has many lines, one for every event, arranged in four column from left to right as follows:
		*  timestamp of event in microseconds
		*  x coordinate of event
		*  y coordinate of event
		*  polarity of event (-1 or 1)

	Arguments:
		* ``filename`` (``string``): path to the binary file.

	Usage:

	>>> TD = spikeFileIO.read2Dspikes(file_path)
	'''
	tEvent,xEvent,yEvent,pEvent = np.loadtxt(filename, dtype=int, unpack=True)	
	return event(xEvent, yEvent, pEvent, tEvent/1000)	# convert spike times to ms



# Define dataset module
class IBMGestureDataset(Dataset):
    '''
    This class provides a way to read spike event.

    Members:
      	* ``datasetPath``           : path of the directory in which we have the dataset.
	* ``sampleFile_car``        : path of file of cars sample.
	* ``sampleFile_background`` : path of file of background sample.
	* ``samplingTime``          : sampling rate at which perform accumulation (ms). 
      	* ``sampleLength``          : length of the sample extract from the entire stream (ms).
	* ``shift_x``               : value of pixel to shift the attention window on x coordinate. Default: 0.
	* ``shift_y``               : value of pixel to shift the attention window on y coordinate. Default: 0.
	* ``att_window``            : x and y size of attention window. Default: [80 80].
    Usage:

    >>> trainingSet = IBMGestureDataset(datasetPath=data_path_train, 
									sampleFile_car  ='./../N_cars/car_train.txt',
									sampleFile_background  ='./../N_cars/background_train.txt',
									samplingTime=1,
									sampleLength=10,
									shift_x=0,
 									shift_y=0, 
									att_window=[100,100]
									)

    '''




    def __init__(self, datasetPath, sampleFile_car, sampleFile_background, samplingTime, sampleLength, shift_x=0, shift_y=0, att_window=[80,80]):        
        self.path = datasetPath 
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.att_window = att_window
        samples_car = np.loadtxt(sampleFile_car, dtype=str, delimiter='\n', usecols=[0])
        samples_background = np.loadtxt(sampleFile_background, dtype=str, delimiter='\n', usecols=[0])
        
        # prepare the list of the entire dataset and annoted the classes
        sample=np.concatenate((samples_car,samples_background),axis=None)        
        classification = np.concatenate((np.ones(len(samples_car)),np.zeros(len(samples_background))),axis=None)  
        
        #shuffle samples
        indices= np.arange(sample.shape[0])
        np.random.shuffle(indices)
        sample=sample[indices]
        classification=classification[indices]
        self.samples = [sample,classification]
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        # Read input files and label
        inputIndex  = self.samples[0][index]
        classLabel  = int(self.samples[1][index])

        # Read input spike from input files (read only the part selected by sampleLength)
        inputSpikes = read2Dspikes(
                        self.path + str(inputIndex.item()) + '.dat'
                        ).toSpikeTensor(torch.zeros((2,self.att_window[0],self.att_window[1],self.nTimeBins)),
                        samplingTime=self.samplingTime, shift_x=self.shift_x, shift_y= self.shift_y)
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((2, 1,1,1))
        desiredClass[classLabel,...] = 1
        
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return len(self.samples[0])
