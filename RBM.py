import torch
import math
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import copy

'''
30 06 2022
Questo codice replica in Pytorch il codice Octave_CPU
il codice implementa bene una RBM monostrato. Va fatto lavoro 
per implementare una DBN multistrato. Il nome della attuale classe 
è perciò fuorviante
'''
def relu(x, upper_bound = 0):
    r = torch.maximum(torch.zeros_like(x), x)
    if upper_bound>0:
        tetto = torch.zeros_like(x)+upper_bound
        r = torch.minimum(tetto, r)
    #print('min:' +str(torch.min(torch.flatten(r))),'max:' +str(torch.max(torch.flatten(r))))
    return r


class RBM():
    def __init__(self,
                layersize = 1000, #size of the hidden layer
                maxepochs   = 10, # unsupervised learning epochs
                batchsize   = 125, # mini-batch size
                sparsity       = 1, # set to 1 to encourage sparsity on third layer
                spars_factor   = 0.05, # how much sparsity?
                epsilonw       = 0.1, # learning rate (weights)
                epsilonvb      = 0.1, # learning rate (visible biases)
                epsilonhb      = 0.1, # learning rate (hidden biases)
                weightcost     = 0.0002, # decay factor
                init_momentum  = 0.5, # initial momentum coefficient
                final_momentum = 0.9,
                device ='cuda',
                Num_classes = 10,
                Hidden_mode = 'binary', # alternative:ReLU
                Visible_mode='continous'): #alternative: binary

        self.layersize = layersize
        self.maxepochs   = maxepochs
        self.batchsize   = batchsize
        self.sparsity       = sparsity
        self.spars_factor   = spars_factor
        if Hidden_mode=='binary':
            self.epsilonw       = epsilonw
        else:
            self.epsilonw       = 0.0075
        self.epsilonw       = epsilonw
        self.epsilonvb      = epsilonvb
        self.epsilonhb      = epsilonhb
        self.weightcost     = weightcost
        self.init_momentum  = init_momentum
        self.final_momentum = final_momentum
        self.DEVICE = device
        self.Num_classes = Num_classes
        self.Visible_mode = Visible_mode
        self.Hidden_mode = Hidden_mode


    def train(self, dataset, train_labels):

        tensor_x = dataset.type(torch.FloatTensor).to(self.DEVICE) # transform to torch tensors
        tensor_y = train_labels.type(torch.FloatTensor).to(self.DEVICE)
        _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
        _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=self.batchsize,drop_last = True) # create your dataloader

        '''
        The drop_last=True parameter ignores the last batch (when the number of examples in your dataset is not
        divisible by your batch_size) while drop_last=False will make the last batch smaller than your batch_size
        see also https://pytorch.org/docs/1.3.0/data.html#torch.utils.data.DataLoader

        to check the dataloader structure
        for batch_idx, samples in enumerate(_dataloader):
            print(batch_idx, samples)
        '''

        self.err = torch.FloatTensor(self.maxepochs).to(self.DEVICE) #this tensor will contain the error as a function of the training epoch

        # parameter initialization
        numhid  = self.layersize
        numcases = self.batchsize
        numdims = tensor_x.size()[1]*tensor_x.size()[2]
        numbatches =math.floor(tensor_x.size()[0]/self.batchsize)
        self.vishid       = 0.1*torch.randn(numdims, numhid).to(self.DEVICE)
        self.hidbiases    = torch.zeros(1,numhid).to(self.DEVICE)
        self.visbiases    = torch.zeros(1,numdims).to(self.DEVICE)
        self.vishidinc    = torch.zeros(numdims, numhid).to(self.DEVICE)
        self.hidbiasinc   = torch.zeros(1,numhid).to(self.DEVICE)
        self.visbiasinc   = torch.zeros(1,numdims).to(self.DEVICE)
        batchposhidprobs = torch.zeros(self.batchsize, numhid, numbatches).to(self.DEVICE)

        for epoch in range (self.maxepochs): #for every epoch...
            errsum = 0
            for mb, samples in enumerate(_dataloader): #for every batch...
                data_mb = samples[0]
                data_mb = data_mb.view(len(data_mb) , numdims)
                err, poshidprobs = self.train_RBM(data_mb,numcases,epoch)
                errsum = errsum + err
                if epoch == self.maxepochs:
                    batchposhidprobs[:, :, mb] = poshidprobs
            self.err[epoch] = errsum; 
    
    def train_RBM(self,data_mb,numcases, epoch):
        momentum = self.init_momentum
        #START POSITIVE PHASE
        H_act = torch.matmul(data_mb,self.vishid)
        H_act = torch.add(H_act, self.hidbiases) #W.x + c
        if self.Hidden_mode == 'binary':
            poshidprobs = torch.sigmoid(H_act)
        else:
            poshidprobs = H_act
        posprods     = torch.matmul(torch.transpose(data_mb, 0, 1), poshidprobs)
        poshidact    = torch.sum(poshidprobs,0)
        posvisact    = torch.sum(data_mb,0)
        #END OF POSITIVE PHASE
        if self.Hidden_mode == 'binary':
            poshidstates = torch.bernoulli(poshidprobs)
        elif self.Hidden_mode == 'gaussian':
            noise = torch.randn(poshidprobs.shape,device='cuda')
            poshidstates = poshidprobs+noise
        elif self.Hidden_mode == 'ReLU':
            noise = torch.randn(poshidprobs.shape,device='cuda')
            poshidstates = relu(poshidprobs+noise)
        elif self.Hidden_mode == 'NReLU':
            noise = torch.randn(poshidprobs.shape,device='cuda')*torch.sigmoid(poshidprobs)
            poshidstates = relu(poshidprobs+noise)

        #START NEGATIVE PHASE
        N_act = torch.matmul(poshidstates,torch.transpose(self.vishid, 0, 1))
        N_act = torch.add(N_act, self.visbiases) #W.x + c
        negdata = torch.sigmoid(N_act)
        N2_act = torch.matmul(negdata,self.vishid)
        N2_act = torch.add(N2_act, self.hidbiases) #W.x + c
        if self.Hidden_mode == 'binary':
            neghidprobs = torch.sigmoid(N2_act)
        else:
            neghidprobs = N2_act
        negprods    = torch.matmul(torch.transpose(negdata, 0, 1), neghidprobs)
        neghidact   = torch.sum(neghidprobs,0)
        negvisact   = torch.sum(negdata,0)
        #END OF NEGATIVE PHASE

        err = math.sqrt(torch.sum(torch.sum(torch.square(data_mb - negdata),0)).item())

        if epoch > 5:
            momentum = self.final_momentum

        # UPDATE WEIGHTS AND BIASES
        # non controllati bene quanto il codice precedente
        self.vishidinc  = momentum * self.vishidinc  + self.epsilonw*( (posprods-negprods)/numcases - (self.weightcost * self.vishid))
        self.visbiasinc = momentum * self.visbiasinc + (self.epsilonvb/numcases)*(posvisact-negvisact)
        self.hidbiasinc = momentum * self.hidbiasinc + (self.epsilonhb/numcases)*(poshidact-neghidact)
        self.vishid     = self.vishid + self.vishidinc
        self.visbiases  = self.visbiases + self.visbiasinc
        self.hidbiases  = self.hidbiases + self.hidbiasinc
        # END OF UPDATES

        return err, poshidprobs
    


    def energy_f(self, hid_states, vis_states):

        sum_h_v_W = torch.zeros(hid_states.size()[0],1).to(self.DEVICE)
        m1=torch.matmul(vis_states,self.vishid)
        m2 = torch.matmul(m1,torch.transpose(hid_states,0,1))
        sum_h_v_W = torch.diagonal(m2*torch.eye(m2.size()[0]).to(self.DEVICE))
        state_energy = -torch.matmul(vis_states,torch.transpose(self.visbiases,0,1)) - torch.matmul(hid_states,torch.transpose(self.hidbiases,0,1)) -sum_h_v_W.unsqueeze(1)
        
        return state_energy


    def save_model(self):
        #lavora con drive
        V = 'V'+self.Visible_mode[0]
        H = 'H'+self.Hidden_mode[0]
        if self.Hidden_mode == 'NReLU':
            H=H+self.Hidden_mode[1]

        self.filename = 'OctaveCPU_RBM'+ str(self.maxepochs)+'_'+V+H+'_nr_steps'+str(self.maxepochs)

        object = self
 

        from google.colab import drive
        drive.mount('/content/gdrive')

        save_path = "/content/gdrive/My Drive/"+self.filename

        try:
            os.mkdir(save_path)
        except:
            print("Folder already found")

        Filename = save_path +'/'+ self.filename + '.pkl'

        with open(Filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
