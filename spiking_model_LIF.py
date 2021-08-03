import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = None # neuronal threshold 
decay = None # decay constants
num_classes = None
batch_size  = None
learning_rate = None
lr_decay=None
num_epochs = 200 # max epoch to run

kernel_init = None #variable used to calculate the size of the layers of the network  
net_parm=[]        # save the parameter of every layer


class ActFun(torch.autograd.Function):
    """
    Define approximate firing function
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        lens = thresh # hyper-parameters of approximate function: we set it equal to neuronal threshold
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

 
def mem_update(ops, x, mem, spike):
    """
    Membrane potential update.
    The following code represent the LIF model for neurons

    """
    #print(ops(x).size())
    #print(spike.size())
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

def optimizeWeightBits(weight, scale_factor_full_precision=1):
    '''
    This function quantize the weights in order to represent them in this form:
    
    weight_Loihi = weight * 2^weightExponent

    where weight is represented by 8 bits as in the Loihi chip

    Arguments;
        * ``weights``: full precision weight tensor.
        * ``scale_factor_full_precision``: scale factor apply directly on full precision weights befor calculate weightExponent and weightBits
                                           (if you change this parameter change also the threashold accordingly) Default: 1 

    Usage:

    >>> weights_Loihi = optimizeWeightBits(full_precision_Weights, scale_factor_full_precision=32)
    '''                
    maxWeight = np.max(weight)
    minWeight = np.min(weight)
        
    isSigned = 1 # we suppose signed weights for every different layer
        
    posScale = 127/maxWeight
    negScale = -128/minWeight
        
    scale = np.min([posScale, negScale]) 
        
    scaleBits = np.floor(np.log2(scale)) + isSigned
        
                
    # define the weightExponent
    weightExponent = -scaleBits

    # discretize and truncate the weights so it is almost the same as we represent it on a maximum of 8 bit (coherently with Loihi Chip expressions)  
    weight=np.trunc(weight*2**scaleBits)
        
    weight_Loihi=weight*2**-scaleBits
        
    return weight_Loihi


def genLoihiParams(net):    
    
    '''
    This function save the weights of the network, so we can load them with putWeight(net).
    
    Arguments;
        * ``net``: network without weights.        

    Usage:

    >>> genLoihiParams(weighted_net)
    '''
    
    fc_index=0
    conv_index=0
    kernel=kernel_init[:]
    for net_p in net_parm:
        if len(net_p) == 6:
            conv_weights=net.conv[conv_index].weight.cpu().data.numpy()
            np.save('Trained_'+str(kernel[0])+'/conv'+str(conv_index)+'.npy'  , conv_weights)
            conv_index=conv_index+1
        elif len(net_p) ==2:
            fc_weights=net.fc[fc_index].weight.cpu().data.numpy()
            np.save('Trained_'+str(kernel[0])+'/fc'+str(fc_index)+'.npy'  , fc_weights)
            fc_index=fc_index+1


def putWeight(net):
    '''
    This function load weights, previously saved, on the network.
    This is similar to the load of an entire Pytorch network, but it is more coherent with the implementation on the Loihi

    Arguments;
        * ``net``: network without weights.        

    Usage:

    >>> weighted_net = putWeight(unweighted_net)
    '''
    
    fc_index=0
    conv_index=0
    kernel=kernel_init[:]
    for net_p in net_parm:
        if len(net_p) == 6:
            net.conv[conv_index].weight.data=torch.nn.Parameter(torch.from_numpy(optimizeWeightBits(np.load(file_net+'/conv'+str(conv_index)+'.npy')))) 
            conv_index=conv_index+1
        elif len(net_p) ==2:
            net.fc[fc_index].weight.data=torch.nn.Parameter(torch.from_numpy(optimizeWeightBits(np.load(file_net+'/fc'+str(fc_index)+'.npy'))))
            fc_index=fc_index+1
    return net


def lr_scheduler(optimizer, epoch, lr_decay_epoch=10, lr_decay_value=0.1):
    '''
    This function provide the learning rate decay by a factor of lr_decay_value every lr_decay_epoch epochs

    Arguments;
        * ``optimizer``: optimizer that we want to change.
        * ``epoch``: current epoch.
        * ``lr_decay_epoch``: number of epoch after we implement decay.
        * ``lr_decay_value``: decay factor.

    Usage:

    >>> optimizer = lr_scheduler(optimizer, epoch, lr_decay_epoch, lr_decay_value)
    '''

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] *lr_decay_value
    return optimizer

class SCNN(nn.Module):
    '''
    This class provides the network implemenattion with Pytorch.

    Members:
         * ``conv`` : list of convolution layers.
         * ``fc``   : list of fully connected or dense layers.
         * ``drop`` : list of dropout layers.
         
    Usage:
    >>> net = SCNN()
    '''
    def __init__(self):
        '''
        This function prepare the module of the network inserting the rights parameters

        Variables;
            * ``conv`` : list of convolution layers.
            * ``fc``   : list of fully connected or dense layers.
            * ``drop`` : list of dropout layers.

        ''' 
        super(SCNN, self).__init__()
        
        self.conv=nn.ModuleList()
        self.fc=nn.ModuleList()
        self.drop=nn.ModuleList()
        # we scroll the net_parm and build every module of the network
        for i,net in enumerate(net_parm):
            if len(net) == 6:
                in_planes, out_planes, stride, padding, kernel_size, groups = net
                self.conv.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
                
                
            elif len(net) == 2:
                in_planes, out_planes = net 
                print(in_planes, out_planes)
                self.fc.append(nn.Linear(in_planes, out_planes, bias=False))
                
            elif len(net) ==1 and net[0]>=10:
                self.drop.append(nn.Dropout(p=net[0]/100))
        
    def forward(self, input, time_window = 20):
        '''
        This function represent the forward pass of the network

        Variables;
            * ``c_mem``	     : list that represent the membrane potential of every neuron used in convolution layers.
            * ``c_spike``    : list that represent the spikes firing from every neuron used in convolution layers.
            * ``h_mem``      : list that represent the membrane potential of every neuron used in dense layers.
            * ``h_spike``    : list that represent the spikes firing from every neuron used in dense layers.
            * ``h_sumspike`` : list that represent the sum of the spikes firing from every neuron used in dense layers.

        '''

        c_mem=[]
        c_spike=[]
        h_mem=[]
        h_spike=[]
        h_sumspike=[]
       
        kernel= kernel_init[:]

        # scroll the net parameters and associate neuron runtime information to the involved layers 
        for net_p in net_parm:
            
            if len(net_p)==6:
                # stride implementation
                kernel[0]=-(-kernel[0]//net_p[2]) 
                kernel[1]=-(-kernel[1]//net_p[2])
                
                # padding implemenattion
                kernel[0]=kernel[0]-2*(net_p[3]==0)
                kernel[1]=kernel[1]-2*(net_p[3]==0)
                c_mem.append(torch.zeros(batch_size, net_p[1], kernel[0], kernel[1], device=device))
                c_spike.append(torch.zeros(batch_size, net_p[1], kernel[0], kernel[1], device=device))
                
                
            elif len(net_p)==2:
                h_mem.append(torch.zeros(batch_size, net_p[1], device=device))
                h_spike.append(torch.zeros(batch_size, net_p[1], device=device))
                h_sumspike.append(torch.zeros(batch_size, net_p[1], device=device))
            elif len(net_p)==1 and net_p[0]<10:
                # if we set ceil_mode=True in F.avg_pool2d we must use the expression on the right
                # remember that for the Loihi implementation with DNN module ceil_mode must be False 
                kernel[0]=int(kernel[0]/net_p[0])#int(np.ceil(kernel[0]/net_p[0]))
                kernel[1]=int(kernel[1]/net_p[0])#int(np.ceil(kernel[1]/net_p[0]))
                
        for step in range(time_window): # simulation time steps
            x = input 
            
            x=x.float()
            # indexes to track the passage on convolution, dense and dropout layers 
            c=0
            l=0
            drop=0
            
            for net_p in net_parm:
                
                if len(net_p)==6:
                    c_mem[c],c_spike[c]=mem_update(self.conv[c].to(device), x, c_mem[c], c_spike[c])
                    x=c_spike[c]
                    c=c+1
                elif len(net_p)==2:
                    if l==0:
                        x = x.view(batch_size, -1)
                    h_mem[l], h_spike[l] = mem_update(self.fc[l].to(device), x, h_mem[l], h_spike[l])
                    x=h_spike[l]
                    h_sumspike[l] += h_spike[l]
                    l=l+1
                elif len(net_p)==1 and net_p[0]<10:
                    x = F.avg_pool2d(x,net_p[0],ceil_mode=False)
                    
                    	
                elif len(net_p)==1 and net_p[0]>=10:
                    x = self.drop[drop](x).to(device)
                    drop = drop+1 
        outputs = h_sumspike[-1] / time_window
        return outputs


def initialize_model(filename_net, thresh_f, decay_f, num_classes_f, batch_size_f, learning_rate_f, kernel_init_f) :
    '''
    This function initialize the model with the information passed by arguments at command line

    Arguments;
        * ``filename_net``: name of the file that represent the network we would use.
                            In this file the type of layer and its parameters are defined as follow:
                            -conv2D: channel_in channel_out stride padding kernel_size(the same for x and y) group
                            -avgPool2D: stride(number<10)
                            -dropout: percentage(number>=10)
                            -Dense: input_size output_size

        * ``thresh_f``: neurons threashold.
        * ``decay_f``: decay factor of the membrane of neurons.
        * ``num_classes_f``: number of classes.
        * ``batch_size_f``: batch size used.
        * ``learning_rate_f``: learning rate used.
        * ``kernel_init_f``: size of inputs images.
        
    Usage:

    >>> initialize_model(filename_net, thresh_f, decay_f, num_classes_f, batch_size_f, learning_rate_f, kernel_init_f)
    '''

    with open(filename_net) as openfileobject:
        for line in openfileobject:
            vect_line=[int(s) for s in line.split() if s.isdigit()]
            net_parm.append(vect_line)
    global file_net
    global thresh
    global decay
    global num_classes
    global batch_size
    global learning_rate
    global kernel_init
	
    file_net = filename_net[6:-4]
    thresh = thresh_f
    decay = decay_f
    num_classes = num_classes_f
    batch_size = batch_size_f
    learning_rate = learning_rate_f
    kernel_init = kernel_init_f
	

