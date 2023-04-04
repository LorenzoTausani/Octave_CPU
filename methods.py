import torch
import math
import copy

def label_biasing(model, on_digits=1, topk = 149):

        # aim of this function is to implement the label biasing procedure described in
        # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full
        

 
        # Now i set the label vector from which i will obtain the hidden layer of interest 
        Biasing_vec = torch.zeros (model.Num_classes,1, device = model.DEVICE)
        Biasing_vec[on_digits] = 1

        #I compute the biased hidden vector as the matmul of the trasposed weights_inv and the biasing vec. gen_hidden will have size (Hidden layer size x 1)
        gen_hidden= torch.matmul(torch.transpose(model.weights_inv,0,1), Biasing_vec)

        if topk>-1: #ATTENZIONE: label biasing con più di una label attiva (e.g. on_digits=[4,6]) funziona UNICAMENTE con topk>-1 (i.e. attivando le top k unità piu attive e silenziando le altre)
        #In caso contrario da errore CUDA non meglio specificato
          H = torch.zeros_like(gen_hidden, device = model.DEVICE) #crea un array vuoto della forma di gen_hidden
          for c in range(gen_hidden.shape[1]): # per ciascun esempio di gen_hidden...
            top_indices = torch.topk(gen_hidden[:,c], k=topk).indices # computa gli indici più attivati
            H[top_indices,c] = 1 #setta gli indici più attivi a 1
          gen_hidden = H # gen_hidden ha ora valori binari (1 e 0)



        return gen_hidden

def generate_from_hidden(model, input_hid_prob , nr_gen_steps, temperature=1, consider_top_k_units = 1000, include_energy = 0):

    if isinstance(temperature, list): #if we have a list of temperatures...
        n_times = math.ceil(nr_gen_steps/len(temperature))
        temperature = temperature*n_times

    #input_hid_prob has size Nr_hidden_units x num_cases. Therefore i transpose it
    input_hid_prob = torch.transpose(input_hid_prob,0,1)

    numcases = input_hid_prob.size()[0] #numbers of samples to generate

    hidden_layer_size = input_hid_prob.size()[1]

    hid_prob = torch.zeros(numcases,hidden_layer_size, nr_gen_steps, device=model.DEVICE)
    hid_states = torch.zeros(numcases,hidden_layer_size, nr_gen_steps, device=model.DEVICE)

    vis_prob = torch.zeros(numcases, 784, nr_gen_steps, device=model.DEVICE)
    vis_states = torch.zeros(numcases ,784, nr_gen_steps, device=model.DEVICE)

    Energy_matrix = torch.zeros(numcases, nr_gen_steps, device=model.DEVICE)

    for step in range(0,nr_gen_steps):
        
        if step>0: #if we are after the 1st gen step...
            
            hid_activation = torch.matmul(vis_states[:,:,step-1], model.vishid) + model.hidbiases # hid_act = V(s-1)*W + bH
            
            # Pass the activation in a sigmoid
            if temperature==1:
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation)
            elif isinstance(temperature, list):
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature[step])
            else:
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature)
        else: #if it is the 1st step of generation...
            hid_prob[:,:,step]  = input_hid_prob #the hidden probability is the one in the input

        if model.Hidden_mode=='binary': #If the hidden layer is set to be binary...

            if consider_top_k_units < hidden_layer_size: #if we want to consider just the top k units in the hidden layer...

                #get the indices (idxs) of the smallest (i.e., least probable) units in hid_prob[:,:,step]
                #Idxs size: numcases x (hidden_layer_size - consider_top_k_units)
                vs, idxs = torch.topk(hid_prob[:,:,step], (hidden_layer_size - consider_top_k_units), largest = False) 

                b = copy.deepcopy(hid_prob[:,:,step]) # b is a deepcopy of the original hid_prob[:,:,step]


                for row in range(numcases): # for every sample of b...
                    b[row, idxs[row,:]]=0 #set the indices of the smallest (hidden_layer_size - consider_top_k_units) units to 0

                hid_states[:,:,step] = torch.bernoulli(b) #do the bernoullian sampling
            else:
                hid_states[:,:,step] = torch.bernoulli(hid_prob[:,:,step]) #do the bernoullian sampling
        else:
            hid_states[:,:,step] = hid_prob[:,:,step] #if the hidden layer is set to be continous, avoid the bernoullian sampling

        vis_activation = torch.matmul(hid_states[:,:,step],torch.transpose(model.vishid, 0, 1)) + model.visbiases #V(s) = H(s)*W + bV
        
        #pass the visible activation through the sigmoid to obtain the visible probabilities
        if temperature==1:
            vis_prob[:,:,step]  = torch.sigmoid(vis_activation)
        elif isinstance(temperature, list):
            vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature[step])
        else:
            vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature)


        if model.Visible_mode=='binary': # do the bernoullian sampling if Visible_mode is set to be binary
            vis_states[:,:,step] = torch.bernoulli(vis_prob[:,:,step])
        elif model.Visible_mode=='continous':
            vis_states[:,:,step] = vis_prob[:,:,step]

        if  include_energy == 1: #if you want also to compute energy...
            state_energy = model.energy_f(hid_states[:,:,step], vis_states[:,:,step])
            Energy_matrix[:,step] = state_energy[:,0]


    result_dict = dict(); 
    result_dict['hid_states'] = hid_states
    result_dict['vis_states'] = vis_states
    result_dict['Energy_matrix'] = Energy_matrix
    result_dict['hid_prob'] = hid_prob
    result_dict['vis_prob'] = vis_prob


    return result_dict