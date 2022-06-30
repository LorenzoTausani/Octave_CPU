import torch
import math
import numpy as np

'''
30 06 2022
Questo codice replica in Pytorch il codice Octave_CPU
il codice implementa bene una RBM monostrato. Va fatto lavoro 
per implementare una DBN multistrato. Il nome della attuale classe 
è perciò fuorviante
'''


class DBN():
    def __init__(self,
                layersize = [1000],
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
                device ='cuda'): # momentum coefficient

        self.nlayers = len(layersize)
        self.rbm_layers =[] #decidi che farci
        self.layersize = layersize
        self.maxepochs   = maxepochs
        self.batchsize   = batchsize
        self.sparsity       = sparsity
        self.spars_factor   = spars_factor
        self.epsilonw       = epsilonw
        self.epsilonvb      = epsilonvb
        self.epsilonhb      = epsilonhb
        self.weightcost     = weightcost
        self.init_momentum  = init_momentum
        self.final_momentum = final_momentum
        self.DEVICE = device



    def train(self, dataset, train_labels):

        tensor_x = dataset.type(torch.FloatTensor).to(self.DEVICE) # transform to torch tensors
        tensor_y = train_labels.type(torch.FloatTensor).to(self.DEVICE)
        _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
        _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=self.batchsize,drop_last = True) # create your dataloader

        self.err = torch.FloatTensor(self.maxepochs,self.nlayers).to(self.DEVICE)

        for layer in range(self.nlayers):
            print('Training layer %d...\n', layer)

            if layer == 0:
                data = dataset
            else:
                data  = batchposhidprobs #da definire

            # initialize weights and biases
            numhid  = self.layersize[layer]
            # forse da cambiare
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

            for epoch in range (self.maxepochs):
                errsum = 0
                for mb, samples in enumerate(_dataloader):
                    data_mb = samples[0]
                    data_mb = data_mb.view(len(data_mb) , numdims)
                    err, poshidprobs = self.train_RBM(data_mb,numcases,epoch)
                    errsum = errsum + err
                    if epoch == self.maxepochs:
                        batchposhidprobs[:, :, mb] = poshidprobs
                    #sono arrivato qui 29/6 h 19
                    if self.sparsity and (layer == 2):
                        poshidact = torch.sum(poshidprobs,0)
                        Q = poshidact/self.batchsize
                        if torch.mean(Q) > self.spars_factor:
                            hidbiases = hidbiases - self.epsilonhb*(Q-self.spars_factor)
                self.err[epoch, layer] = errsum; 
        

    def train_RBM(self,data_mb,numcases, epoch):
        momentum = self.init_momentum
        #START POSITIVE PHASE
        H_act = torch.matmul(data_mb,self.vishid)
        H_act = torch.add(H_act, self.hidbiases) #W.x + c
        poshidprobs = torch.sigmoid(H_act)
        posprods     = torch.matmul(torch.transpose(data_mb, 0, 1), poshidprobs)
        poshidact    = torch.sum(poshidprobs,0)
        posvisact    = torch.sum(data_mb,0)
        #END OF POSITIVE PHASE
        poshidstates = torch.bernoulli(poshidprobs)

        #START NEGATIVE PHASE
        N_act = torch.matmul(poshidstates,torch.transpose(self.vishid, 0, 1))
        N_act = torch.add(N_act, self.visbiases) #W.x + c
        negdata = torch.sigmoid(N_act)
        N2_act = torch.matmul(negdata,self.vishid)
        N2_act = torch.add(N2_act, self.hidbiases) #W.x + c
        neghidprobs = torch.sigmoid(N2_act)
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

    def reconstruct(input_data, rbm_mnist, nr_steps):
        numcases = input_data.size()[0]
        vector_size = input_data.size()[1]*input_data.size()[2]
        input_data =  input_data.view(len(input_data) , vector_size)
        hid_prob = torch.zeros(numcases,rbm_mnist.layersize[0],nr_steps).to(rbm_mnist.DEVICE)
        hid_states = torch.zeros(numcases,rbm_mnist.layersize[0],nr_steps).to(rbm_mnist.DEVICE)

        vis_prob = torch.zeros(numcases,vector_size, nr_steps).to(rbm_mnist.DEVICE)
        vis_states = torch.zeros(numcases,vector_size, nr_steps).to(rbm_mnist.DEVICE)

        for step in range(0,nr_steps):
            print(step)
            if step==0:
                hid_activation = torch.matmul(input_data,rbm_mnist.vishid) + rbm_mnist.hidbiases
                print("eccomi if")
            else:
                hid_activation = torch.matmul(vis_states[:,:,step-1],rbm_mnist.vishid) + rbm_mnist.hidbiases
                print("eccomi else")

            hid_prob[:,:,step]  = torch.sigmoid(hid_activation)
            
            hid_states[:,:,step] = torch.bernoulli(hid_prob[:,:,step])

            vis_activation = torch.matmul(hid_states[:,:,step],torch.transpose(rbm_mnist.vishid, 0, 1)) + rbm_mnist.visbiases

            vis_prob[:,:,step]  = torch.sigmoid(vis_activation)

            vis_states[:,:,step] = torch.bernoulli(vis_prob[:,:,step])

            #manca energia

        return vis_states        







