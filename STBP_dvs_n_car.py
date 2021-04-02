from __future__ import print_function
import argparse
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model_LIF import*
from N_cars_dataset import*


#init value for python script

parser=argparse.ArgumentParser()
parser.add_argument('--filenet', type=str, dest='filename_net')
parser.add_argument('--fileresult', type=str, default='result.txt', dest='filename_result')
parser.add_argument('--sample_time', type=float, default=1, dest='sample_time')
parser.add_argument('--sample_length', type=float, default=10, dest='sample_length')
parser.add_argument('--batch_size', type=int, default=40, dest='batch_size')
parser.add_argument('--lr', type=float, default=1e-3, dest='lr')
parser.add_argument('--lr_decay_epoch', type=int, default=20, dest='lr_decay')
parser.add_argument('--lr_decay_value', type=float, default=0.5, dest='lr_decay_value')
parser.add_argument('--threshold', type=float, default=0.4, dest='thresh')
parser.add_argument('--n_decay', type=float, default=0.2, dest='n_decay') #decay constant
parser.add_argument('--att_window', type=int, nargs=4, dest='att_window')
parser.add_argument('--weight_decay', type=float, default=0, dest='weight_decay') #L2regularizzation


args = parser.parse_args()

# initialize spiking model and network
initialize_model(args.filename_net, args.thresh, args.n_decay, 2, args.batch_size, args.lr, kernel_init_f=[args.att_window[0], args.att_window[1]])

batch_size= args.batch_size


data_path_train =  './'  #todo: input your data path for train dataset if not write in train files (car_train.txt and background_train.txt)
data_path_test =  './'   #todo: input your data path for test dataset if not write in test files (car_test.txt and background_test.txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samplingTime = args.sample_time
sampleLength = args.sample_length
filename_result = args.filename_result

# instantiate the train dataset and use the DataLoader function to give samples to the network 
trainingSet = IBMGestureDataset(datasetPath=data_path_train, 
									sampleFile_car  ='./N_cars/car_train.txt',
									sampleFile_background  ='./N_cars/background_train.txt',
									samplingTime=samplingTime,
									sampleLength=sampleLength,
									shift_x=args.att_window[2],
 									shift_y=args.att_window[3], 
									att_window=[args.att_window[0],	args.att_window[1]])

train_loader  = DataLoader(dataset=trainingSet, batch_size=batch_size, shuffle=True, num_workers=10)

# instantiate the test dataset and use the DataLoader function to give samples to the network 
testingSet = IBMGestureDataset(datasetPath=data_path_test, 
									sampleFile_car  ='./N_cars/car_test.txt',
									sampleFile_background  ='./N_cars/background_test.txt',
									samplingTime=samplingTime,
									sampleLength=sampleLength,
									shift_x=args.att_window[2],
 									shift_y=args.att_window[3], 
									att_window=[args.att_window[0],	args.att_window[1]])
test_loader = DataLoader(dataset=testingSet, batch_size=batch_size, shuffle=True, num_workers=10)

# create and open the file to write the principle numerical results runtime 
f=open(filename_result, 'w')

# write the principal initialization information 
f.write('batch size: '+str(args.batch_size)+ ' sampling time: '+str(samplingTime)+ ' sampling length: '+str(sampleLength)+ ' filenet: '+str(args.filename_net)+ ' learning rate: '+str(args.lr)+ ' lr decay epoch: '+str(args.lr_decay)+ ' lr decay value: '+str(args.lr_decay_value)+ ' threashold: '+str(args.thresh)+ ' neuron decay constant: '+str(args.n_decay)+ ' attention window: '+str(args.att_window)+ ' weight decay(L2 reg): '+str(args.weight_decay)+'\n')

# define the network and load saved weights
snn = SCNN()
snn = putWeight(snn)
snn.to(device)

# define criterion and optimizer
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False) #L2r

# run the train and test for num_epochs epochs
for epoch in range(num_epochs):
    best_acc_entire_image_test=0
    running_loss = 0
    start_time = time.time()
    
    len_of_sample= len(trainingSet)
    
    snn=snn.train()
    correct_entire_image=0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
    total_entire_image=0   # number of images to predict
    
    for i, (images, labels_,labels) in enumerate(train_loader,0):
        # run only for complete batches
        len_of_sample=len_of_sample-batch_size
        if len_of_sample >= 0:
           snn.zero_grad()
           optimizer.zero_grad()
           images = images.float().to(device)
           first=0
	   # group outputs of the same image of length sampleLength and accumulate the prediction for every samplingTime
           for j in range (0, int(sampleLength/samplingTime)):
              outputs = snn(images[:,:,:,:,j])
              if first==0:
                 _,accumulation=outputs.to(device).max(1)
                 first=first+1
              else:
                 _,predicted=outputs.max(1)
                 accumulation+=predicted
                 
              
              loss = criterion(outputs, labels_[:,:,0,0,0].to(device))
              running_loss += loss.item()
              
              loss.backward()
           optimizer.step()
		
	   # see what is the most predicted class for the image
           accumulation[accumulation<(sampleLength/samplingTime)/2]=0
           accumulation[accumulation>=(sampleLength/samplingTime)/2]=1
           
	   # calculate accuracy on the image of length sampleLength
           total_entire_image += float(labels.size(0))
           correct_entire_image += float(accumulation.eq(labels.to(device)).sum().item())
           acc_entire_image_train=100*correct_entire_image/total_entire_image
        if (i+1)%20 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Accuracy: %.5f'
                    %(epoch+1, num_epochs, i+1, len(trainingSet)//batch_size,running_loss, acc_entire_image_train))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
	
    
    correct = 0 # number of correct decision for each samplingTime 
    total = 0   # number of total samplingTime predictions 
    optimizer = lr_scheduler(optimizer, epoch, args.lr_decay, args.lr_decay_value)
    correct_entire_image=0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
    total_entire_image=0   # number of images of sampleLength length
    with torch.no_grad():
        snn=snn.eval()
        len_of_sample= len(testingSet)
        for batch_idx, (inputs, labels_, targets) in enumerate(test_loader,0):
          # run only for the complete batch size
          len_of_sample=len_of_sample-batch_size
          if len_of_sample >= 0:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            first=0
	    # group outputs of the same image of length sampleLength and accumulate the prediction for every samplingTime
            for j in range (0, int(sampleLength/samplingTime)):
              outputs = snn(inputs[:,:,:,:,j])
              if first==0:
                 _,accumulation=outputs.to(device).max(1)
                 first=first+1
              else:
                 _,pre=outputs.max(1)
                 accumulation+=pre
              
              loss = criterion(outputs, labels_[:,:,0,0,0].to(device))
	      # calculate the prediction at every samplingTime without grouping them in an image of sampleLength length  
              _, predicted = outputs.max(1)
              total += float(targets.size(0))
              correct += float(predicted.eq(targets.to(device)).sum().item())
		
            # see the most predicted class for the image of length sampleLength
            accumulation[accumulation<(sampleLength/samplingTime)/2]=0
            accumulation[accumulation>=(sampleLength/samplingTime)/2]=1
	
	    # calculate accuracy on the image of length sampleLength and at every samplingTime
            total_entire_image += float(targets.size(0))
            correct_entire_image += float(accumulation.eq(targets.to(device)).sum().item())
            acc_entire_image_test=100*correct_entire_image/total_entire_image
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the sampling time streams: %.3f' % (100 * correct / total))
    print('Test Accuracy of the model on the entire test images: %.3f' % (acc_entire_image_test))
    acc = 100. * float(correct) / float(total)
    
    # every epoch save the results 
    if epoch % 1 == 0:
        print(acc)
        print('Saving results..')

        f.write('acc: '+str(acc)+' acc_train: '+str(acc_entire_image_train)+' acc_test: '+str(acc_entire_image_test)+' epoch: '+str(epoch)+'\n')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
	
	# save the network and the weights only if the accuracy on entire images is better than before 
        if epoch>=0 and best_acc_entire_image_test < acc_entire_image_test:
           print('Saving weights and network..')
           best_acc_entire_image_test=acc_entire_image_test
           if not os.path.isdir('checkpoint'):
              os.mkdir('checkpoint')
           torch.save(state, './checkpoint/ckpt' + str(args.att_window[0])+'_ceil' + '.t7')
           genLoihiParams(snn)
