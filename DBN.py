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
                device ='cuda',
                Num_classes = 10,
                Hidden_mode = 'binary',
                bin_threshold = 0.5,
                Visible_mode='continous'): #alternative: binary, leaky_binary

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
        self.Num_classes = Num_classes
        self.Visible_mode = Visible_mode
        self.Hidden_mode = Hidden_mode
        self.bin_threshold = bin_threshold



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
    
    def compute_inverseW_for_lblBiasing(self):
        n_cl = len(torch.unique(self.TRAIN_lbls))
        tr_patterns = torch.squeeze(self.TRAIN_gen_hid_states) #This array contains the 1st hidden state obtained from the reconstruction of all the MNIST training set (size: nr_MNIST_train_ex x Hidden layer size)
        #L is a array of size (model.Num_classes x nr_MNIST_train_ex (10 x 60000)). Each column of it is the one-hot encoded label of the i-th MNIST train example
        L = torch.zeros(n_cl,tr_patterns.shape[0], device = self.DEVICE)
        c=0
        for lbl in self.TRAIN_lbls:
            L[lbl,c]=1
            c=c+1
        #I compute the inverse of the weight matrix of the linear classifier. weights_inv has shape (model.Num_classes x Hidden layer size (10 x 1000))
        weights_inv = torch.transpose(torch.matmul(torch.transpose(tr_patterns,0,1), torch.linalg.pinv(L)), 0, 1)

        self.weights_inv = weights_inv

        return weights_inv

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


    def reconstruct(self, input_data, nr_steps, new_test1_train2_set = 0,lbl_train=[],lbl_test=[], temperature=1, include_energy = 1, consider_top=1000):

        '''
        1 = test, 2 = training
        '''
        if isinstance(temperature, list):
            n_times = math.ceil(nr_steps/len(temperature))
            temperature = temperature*n_times

        
        numcases = input_data.size()[0]
        vector_size = input_data.size()[1]*input_data.size()[2]
        input_data =  input_data.view(len(input_data) , vector_size)
        hid_prob = torch.zeros(numcases,self.layersize[0],nr_steps).to(self.DEVICE)
        hid_states = torch.zeros(numcases,self.layersize[0],nr_steps).to(self.DEVICE)

        vis_prob = torch.zeros(numcases,vector_size, nr_steps).to(self.DEVICE)
        vis_states = torch.zeros(numcases,vector_size, nr_steps).to(self.DEVICE)

        Energy_matrix = torch.zeros(numcases, nr_steps).to(self.DEVICE)

        for step in range(0,nr_steps):
            
            if step==0:
                hid_activation = torch.matmul(input_data,self.vishid) + self.hidbiases
            else:
                hid_activation = torch.matmul(vis_states[:,:,step-1],self.vishid) + self.hidbiases


            if temperature==1:
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation)
            elif isinstance(temperature, list):
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature[step])
            else:
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature)
             
            if self.Hidden_mode=='binary':
                if consider_top<self.layersize[0]:
                    vs, idxs = torch.topk(hid_prob[:,:,step], (self.layersize[0]-consider_top), largest = False) #da testare
                    b = copy.deepcopy(hid_prob[:,:,step])
                    for row in range(b.size()[0]):
                        b[row, idxs[row,:]]=0
                    hid_states[:,:,step] = torch.bernoulli(b)
                else:
                    hid_states[:,:,step] = torch.bernoulli(hid_prob[:,:,step])
            else:
                hid_states[:,:,step] = hid_prob[:,:,step]

            vis_activation = torch.matmul(hid_states[:,:,step],torch.transpose(self.vishid, 0, 1)) + self.visbiases

            if temperature==1:
                vis_prob[:,:,step]  = torch.sigmoid(vis_activation)
            elif isinstance(temperature, list):
                vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature[step])
            else:
                vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature)

            if self.Visible_mode=='binary':
                vis_states[:,:,step] = torch.bernoulli(vis_prob[:,:,step])
            elif self.Visible_mode=='continous':
                vis_states[:,:,step] = vis_prob[:,:,step]
            else: #leaky_binary
                TF_vis_states = vis_prob[:,:,step]>self.bin_threshold
                vis_states[:,:,step] = TF_vis_states.to(torch.float32)

            if  include_energy == 1:
                state_energy = self.energy_f(hid_states[:,:,step], vis_states[:,:,step])
                Energy_matrix[:,step] = state_energy[:,0]

        if new_test1_train2_set == 1:
            self.TEST_gen_hid_states = hid_states
            self.TEST_vis_states = vis_states
            self.TEST_gen_hid_prob = hid_prob
            self.TEST_vis_prob = vis_prob
            self.TEST_lbls = lbl_test
            self.TEST_energy_matrix = Energy_matrix

        elif new_test1_train2_set == 2:
            self.TRAIN_gen_hid_states = hid_states
            self.TRAIN_vis_states = vis_states
            self.TRAIN_gen_hid_prob = hid_prob
            self.TRAIN_vis_prob = vis_prob
            self.TRAIN_lbls = lbl_train

        result_dict = dict(); 
        result_dict['hid_states'] = hid_states
        result_dict['vis_states'] = vis_states
        result_dict['Energy_matrix'] = Energy_matrix
        result_dict['hid_prob'] = hid_prob
        result_dict['vis_prob'] = vis_prob

        return result_dict
        


    def energy_f(self, hid_states, vis_states):

        sum_h_v_W = torch.zeros(hid_states.size()[0],1).to(self.DEVICE)
        m1=torch.matmul(vis_states,self.vishid)
        m2 = torch.matmul(m1,torch.transpose(hid_states,0,1))
        sum_h_v_W = torch.diagonal(m2*torch.eye(m2.size()[0]).to(self.DEVICE))
        state_energy = -torch.matmul(vis_states,torch.transpose(self.visbiases,0,1)) - torch.matmul(hid_states,torch.transpose(self.hidbiases,0,1)) -sum_h_v_W.unsqueeze(1)
        
        return state_energy

    def RBM_perceptron(self, tr_patterns, tr_labels, te_patterns, te_labels):
        '''
        tr_patterns, te_patterns = training and testing data (e.g. hidden states)
        '''
        te_accuracy = 0
        tr_accuracy = 0

        #add biases
        ONES = torch.ones(tr_patterns.size()[0], 1).to(self.DEVICE)
        tr_patterns = torch.cat((torch.squeeze(tr_patterns),ONES), 1)

        #train with pseudo-inverse
        L = torch.zeros(self.Num_classes,len(tr_patterns)).to(self.DEVICE)
        c=0
        for lbl in tr_labels:
            L[lbl,c]=1
            c=c+1

        weights = torch.transpose( torch.matmul(L, torch.linalg.pinv(torch.transpose(tr_patterns,0,1)) ), 0,1)

        # training accuracy
        pred = torch.matmul(tr_patterns,weights)
        max_act = pred.argmax(1) #nota: r del codice originale è tr_labels in questo codice
        acc = max_act == tr_labels
        tr_accuracy = torch.mean(acc.to(torch.float32)).item()

        if not(te_patterns.nelement() == 0):
            # test accuracy
            ONES = torch.ones(te_patterns.size()[0], 1).to(self.DEVICE)
            te_patterns = torch.cat((torch.squeeze(te_patterns),ONES), 1) 
            pred = torch.matmul(te_patterns,weights)
            max_act = pred.argmax(1) #nota: r del codice originale è tr_labels in questo codice
            acc = max_act == te_labels
            te_accuracy = torch.mean(acc.to(torch.float32)).item()

        return tr_accuracy,te_accuracy  

    def stepwise_Cl_accuracy(self):
        te_acc = []
        for i in range(self.TEST_gen_hid_states.size()[2]):
            tr_accuracy,te_accuracy = self.RBM_perceptron(self.TRAIN_gen_hid_states, self.TRAIN_lbls,self.TEST_gen_hid_states[:,:,i], self.TEST_lbls)
            te_acc.append(te_accuracy)
        self.Cl_TEST_step_accuracy = te_acc
        return te_acc    

    def label_biasing(self,nr_steps,temperature=1, row_step=10):
        '''
        scopo di questa funzione è implementare il label biasing descritto in
        https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full
        '''

        tr_patterns = torch.squeeze(self.TRAIN_gen_hid_states)
        n_cl = len(torch.unique(self.TRAIN_lbls))

        L = torch.zeros(n_cl,len(tr_patterns)).to(self.DEVICE)
        c=0
        for lbl in self.TRAIN_lbls:
            L[lbl,c]=1
            c=c+1

        
        weights_inv = torch.transpose(torch.matmul(torch.transpose(tr_patterns,0,1), torch.linalg.pinv(L)), 0, 1)
        lbl_mat=torch.eye(n_cl).to(self.DEVICE)

        gen_hidden_act = torch.matmul(torch.transpose(weights_inv,0,1),lbl_mat)

        '''
        %domanda: passo o no attraverso la sigmoide? ( risposta: empiricamente con
        %il passaggio in sigmoide viene male)
        %hid_prob  = 1./(1 + exp(-gen_hidden_act')); %passo in sigmoide
        %hid_bin = hid_prob > rand(size(hid_prob)); 
        '''

        #hid_bin = torch.bernoulli(torch.transpose(gen_hidden_act,0,1)) #elimina questo passaggio

        vis_activation = torch.matmul(torch.transpose(gen_hidden_act,0,1),torch.transpose(self.vishid, 0, 1)) + self.visbiases #qui passa get_hidden_act
        vis_prob  = torch.sigmoid(vis_activation)
          
        if self.Visible_mode=='binary':
            vis_state = torch.bernoulli(vis_prob)
        elif self.Visible_mode=='continous':
            vis_state = vis_prob
        else: #leaky_binary
            TF_vis_states = vis_prob>self.bin_threshold
            vis_state = TF_vis_states.to(torch.float32)        

        return vis_state, gen_hidden_act

    def cosine_similarity(self, original_data, generated_data, Plot=0, Color='black', Linewidth=1, axis=[]):

        if len(original_data.size())>2: # se vi sono più di due dimensioni, allora faccio il seguente resizing dei dati originali
            vector_size = original_data.size()[1]*original_data.size()[2]
            input_data =  original_data.view(len(original_data) , vector_size)
        else:
            input_data = original_data

        nr_steps = generated_data.size()[2] #il numero di steps di ricostruzione

        input_data_mat = input_data.repeat(nr_steps,1,1) #vedi https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
        #ridimensiono per inputtare a cos dopo
        input_data_mat=torch.transpose(input_data_mat,0,2)
        input_data_mat=torch.transpose(input_data_mat,0,1)

        cos = torch.nn.CosineSimilarity(dim=1)
        c=cos(input_data_mat,generated_data)

        SEM = torch.std(c,0)/math.sqrt(generated_data.size()[0]) #dovrebbe essere corretto dividere per il numero di dati
        SEM = SEM.cpu()
        MEAN = torch.mean(c,0).cpu()

        if Plot ==1:
            
            x = range(1,nr_steps+1)
            
            if axis==[]: #se non è da plottare in un subplot
            
                plt.plot(x, MEAN, c = Color, linewidth=Linewidth)

                plt.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color,
                                alpha=0.3)
                #plt.show()
            
            else: #se invece è da mettere in un subplot
                axis.plot(x, MEAN, c = Color, linewidth=Linewidth)
                axis.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color,
                                alpha=0.3)


        return MEAN, SEM


    def save_model(self):
        #lavora con drive

        try:
            h_test_size = self.TEST_gen_hid_states.shape[0]
            nr_steps = self.TEST_gen_hid_states.shape[2]
        except:
            h_test_size = 0
            nr_steps = 0

        try:
            h_train_size = self.TRAIN_gen_hid_states.shape[0]
        except:
            h_train_size = 0

        if self.Visible_mode=='binary':
           V = 'Vbinary'
        elif self.Visible_mode=='continous':
           V = 'Vcontinous'
        else:
           V = 'VleakyBinary'

        self.filename = 'OctaveCPU_RBM'+ str(self.maxepochs)+'_generated_h_train'+str(h_train_size)+'_generated_h_test'+str(h_test_size)+'nr_steps'+str(nr_steps)+V

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




    def reconstruct_from_hidden(self, input_hid_prob , nr_steps, temperature=1, include_energy = 1,consider_top=1000):

        if isinstance(temperature, list):
            n_times = math.ceil(nr_steps/len(temperature))
            temperature = temperature*n_times



        numcases = input_hid_prob.size()[0]
        vector_size = input_hid_prob.size()[1]*input_hid_prob.size()[2]
        input_hid_prob =  input_hid_prob.view(len(input_hid_prob) , vector_size)
        hid_prob = torch.zeros(numcases,self.layersize[0],nr_steps).to(self.DEVICE)
        hid_states = torch.zeros(numcases,self.layersize[0],nr_steps).to(self.DEVICE)

        vis_prob = torch.zeros(numcases,784, nr_steps).to(self.DEVICE)
        vis_states = torch.zeros(numcases,784, nr_steps).to(self.DEVICE)

        Energy_matrix = torch.zeros(numcases, nr_steps).to(self.DEVICE)

        for step in range(0,nr_steps):
            

            if step>0:
                hid_activation = torch.matmul(vis_states[:,:,step-1],self.vishid) + self.hidbiases

                if temperature==1:
                    hid_prob[:,:,step]  = torch.sigmoid(hid_activation)
                elif isinstance(temperature, list):
                    hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature[step])
                else:
                    hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature)
            else:
                hid_prob[:,:,step]  = input_hid_prob

            if self.Hidden_mode=='binary':
                if consider_top<self.layersize[0]:
                    vs, idxs = torch.topk(hid_prob[:,:,step], (self.layersize[0]-consider_top), largest = False) #da testare
                    b = copy.deepcopy(hid_prob[:,:,step])
                    for row in range(b.size()[0]):
                        b[row, idxs[row,:]]=0
                    hid_states[:,:,step] = torch.bernoulli(b)
                else:
                    hid_states[:,:,step] = torch.bernoulli(hid_prob[:,:,step])
            else:
                hid_states[:,:,step] = hid_prob[:,:,step]

            vis_activation = torch.matmul(hid_states[:,:,step],torch.transpose(self.vishid, 0, 1)) + self.visbiases

            if temperature==1:
                vis_prob[:,:,step]  = torch.sigmoid(vis_activation)
            elif isinstance(temperature, list):
                vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature[step])
            else:
                vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature)


            if self.Visible_mode=='binary':
                vis_states[:,:,step] = torch.bernoulli(vis_prob[:,:,step])
            elif self.Visible_mode=='continous':
                vis_states[:,:,step] = vis_prob[:,:,step]
            else: #leaky_binary
                TF_vis_states = vis_prob[:,:,step]>self.bin_threshold
                vis_states[:,:,step] = TF_vis_states.to(torch.float32)

            if  include_energy == 1:
                state_energy = self.energy_f(hid_states[:,:,step], vis_states[:,:,step])
                Energy_matrix[:,step] = state_energy[:,0]


        result_dict = dict(); 
        result_dict['hid_states'] = hid_states
        result_dict['vis_states'] = vis_states
        result_dict['Energy_matrix'] = Energy_matrix
        result_dict['hid_prob'] = hid_prob
        result_dict['vis_prob'] = vis_prob


        return result_dict













    
    def sliced_wasserstein(self, X, Y, num_proj=10):
        #from https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
        #this refers to https://link.springer.com/article/10.1007/s10851-014-0506-3

        X = X.cpu()
        Y = Y.cpu()
        dim = X.shape[1]
        ests = []
        for _ in range(num_proj):
            # sample uniformly from the unit sphere
            dir = np.random.rand(dim)
            dir /= np.linalg.norm(dir)

            # project the data
            X_proj = X @ dir
            Y_proj = Y @ dir

            # compute 1d wasserstein
            ests.append(wasserstein_distance(X_proj, Y_proj))
        return np.mean(ests)


    def Wasserstein_allGenData(self, original_ds):

        X = original_ds #sample_test_data
        nr_samples = self.TEST_vis_states.size()[0]
        nr_steps = self.TEST_vis_states.size()[2]
        Y = self.TEST_vis_states.view((nr_samples,28,28,nr_steps))

        Wass_mat = torch.zeros(nr_samples, nr_steps)
        for s_idx in range(nr_samples):
            for idx in range(nr_steps):
                Wass_mat[s_idx, idx] = self.sliced_wasserstein(X[s_idx,:,:], Y[s_idx,:,:,idx])

        self.Wass_mat = Wass_mat
        
        return Wass_mat