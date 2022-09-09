import matplotlib.pyplot as plt
from matplotlib import cm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels,batch_norm=False):

        super().__init__()

        conv2_params = {'kernel_size': (3, 3),
                        'stride'     : (1, 1),
                        'padding'   : 1
                        }

        noop = lambda x : x

        self._batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels , **conv2_params)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else noop
        #self.bn1 = nn.GroupNorm(32, out_channels) if batch_norm else noop

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels, **conv2_params)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else noop
        #self.bn2 = nn.GroupNorm(32, out_channels) if batch_norm else noop

        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    @property
    def batch_norm(self):
        return self._batch_norm

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.max_pooling(x)

        return x



class VGG16(nn.Module):

  def __init__(self, input_size, num_classes=11,batch_norm=False):
    super(VGG16, self).__init__()

    self.in_channels,self.in_width,self.in_height = input_size

    self.block_1 = VGGBlock(self.in_channels,64,batch_norm=batch_norm)
    self.block_2 = VGGBlock(64, 128,batch_norm=batch_norm)
    self.block_3 = VGGBlock(128, 256,batch_norm=batch_norm)
    self.block_4 = VGGBlock(256,512,batch_norm=batch_norm)

    self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

  @property
  def input_size(self):
      return self.in_channels,self.in_width,self.in_height

  def forward(self, x):

    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)
    x = self.block_4(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)

    return x


def Classifier_accuracy(input_data, VGG_cl,model, labels=[], Batch_sz= 100, plot=1, dS=30, l_sz=3):
  #input_data = nr_examples x 784 (i.e. image size) x nr_steps
  Cl_pred_matrix = torch.zeros(input_data.size()[0],input_data.size()[2])
  Pred_entropy_mat = torch.zeros(input_data.size()[0],input_data.size()[2])
  digitwise_acc = torch.zeros(model.Num_classes,input_data.size()[2])
  digitwise_avg_entropy = torch.zeros(model.Num_classes,input_data.size()[2])
  digitwise_sem_entropy = torch.zeros(model.Num_classes,input_data.size()[2])
  acc = torch.zeros(input_data.size()[2])
  if labels==[]:
    labels = torch.zeros(input_data.size()[0]).to(model.DEVICE)


  for step in range(input_data.size()[2]):#input_data.size()[2]
    V = input_data[:,:,step]
    V = torch.unsqueeze(V.view((input_data.size()[0],28,28)),1)
    #print(V.size())
    V_int = F.interpolate(V, size=(32, 32), mode='bicubic', align_corners=False)
    #tocca fare a batch, il che è abbastanza una fatica. Ma così mangia tutta la GPU
    _dataset = torch.utils.data.TensorDataset(V_int,labels) # create your datset
    _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=Batch_sz,drop_last = True) # create your dataloader
    
    index = 0
    acc_v = torch.zeros(math.floor(input_data.size()[0]/Batch_sz))
    last_batch_size =Batch_sz*acc_v.size()[0] - input_data.size()[0]
    
    n_it = 0
    for (input, lbls) in _dataloader:
      
      with torch.no_grad():
        pred_vals=VGG_cl(input)
      #return pred_vals
      
      Pred_entropy = torch.distributions.Categorical(probs =pred_vals[:,:10]).entropy()
      Pred_entropy_mat[index:index+Batch_sz,step] = Pred_entropy

      _, inds = torch.max(pred_vals,dim=1)
      Cl_pred_matrix[index:index+Batch_sz,step] = inds
      acc_v[n_it] = torch.sum(inds.to(model.DEVICE)==lbls)/input.size()[0]
      n_it = n_it+1
      index = index+ Batch_sz
    acc[step] = torch.mean(acc_v)
    
    for digit in range(model.Num_classes):
      l = torch.where(labels == digit)
      
      digitwise_avg_entropy[digit,step] = torch.mean(Pred_entropy_mat[l[0],step])
      digitwise_sem_entropy[digit,step] = torch.std(Pred_entropy_mat[l[0],step])/math.sqrt(l[0].size()[0])

      inds_digit = Cl_pred_matrix[l[0],step]
      digitwise_acc[digit,step] = torch.sum(inds_digit.to(model.DEVICE)==labels[l[0]])/l[0].size()[0]

  MEAN_entropy = torch.mean(Pred_entropy_mat,0)
  SEM_entropy = torch.std(Pred_entropy_mat,0)/math.sqrt(input_data.size()[0])


  if plot == 1:
      c=0
      cmap = cm.get_cmap('hsv')
      lbls = range(model.Num_classes)

      figure, axis = plt.subplots(2, 2, figsize=(20,15))
      x = range(1,input_data.size()[2]+1)

      axis[0,0].plot(x, acc, c = 'g', linewidth=l_sz)
        
      axis[0,0].tick_params(axis='x', labelsize= dS)
      axis[0,0].tick_params(axis='y', labelsize= dS)
      axis[0,0].set_ylabel('Accuracy',fontsize=dS)
      axis[0,0].set_ylim([0,1])
      axis[0,0].set_xlabel('Nr. reconstruction steps',fontsize=dS)
      axis[0,0].set_title('VGG accuracy',fontsize=dS)


      axis[0,1].plot(x, MEAN_entropy, c = 'r', linewidth=l_sz)
      axis[0,1].fill_between(x,MEAN_entropy-SEM_entropy, MEAN_entropy+SEM_entropy, color='r',
                      alpha=0.3)
        
      axis[0,1].tick_params(axis='x', labelsize= dS)
      axis[0,1].tick_params(axis='y', labelsize= dS)
      axis[0,1].set_ylabel('Entropy',fontsize=dS)
      axis[0,1].set_ylim([0,1])
      axis[0,1].set_xlabel('Nr. reconstruction steps',fontsize=dS)
      axis[0,1].set_title('Average entropy',fontsize=dS)

      for digit in range(model.Num_classes):
        Color = cmap(c/256) 
        MEAN = digitwise_acc[digit,:].cpu()
        axis[1,0].plot(x, MEAN, c = Color, linewidth=l_sz)
        c = c+25
      
      #axis[1,0].legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #cambia posizione
      axis[1,0].tick_params(axis='x', labelsize= dS)
      axis[1,0].tick_params(axis='y', labelsize= dS)
      axis[1,0].set_ylabel('Accuracy',fontsize=dS)
      axis[1,0].set_ylim([0,1])
      axis[1,0].set_xlabel('Nr. reconstruction steps',fontsize=dS)
      axis[1,0].set_title('VGG accuracy - digitwise',fontsize=dS)
        
      c=0
      for digit in range(model.Num_classes):
        Color = cmap(c/256) 
        MEAN = digitwise_avg_entropy[digit,:].cpu()
        SEM = digitwise_sem_entropy[digit,:].cpu()
        axis[1,1].plot(x, MEAN, c = Color, linewidth=l_sz)
        axis[1,1].fill_between(x,MEAN-SEM, MEAN+SEM, color=Color,
                alpha=0.3)
        c = c+25
      
      axis[1,1].legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #cambia posizione
      axis[1,1].tick_params(axis='x', labelsize= dS)
      axis[1,1].tick_params(axis='y', labelsize= dS)
      axis[1,1].set_ylabel('Entropy',fontsize=dS)
      axis[1,1].set_ylim([0,1])
      axis[1,1].set_xlabel('Nr. reconstruction steps',fontsize=dS)
      axis[1,1].set_title('Entropy - digitwise',fontsize=dS)

      plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.4) 



  result_dict = dict(); 
  result_dict['Cl_pred_matrix'] = Cl_pred_matrix
  result_dict['Cl_accuracy'] = acc
  result_dict['digitwise_acc'] = digitwise_acc
  result_dict['Pred_entropy_mat'] = Pred_entropy_mat
  result_dict['MEAN_entropy'] = MEAN_entropy
   
  return result_dict



def classification_metrics(dict_classifier,model,test_labels, Plot=1, dS = 30):
  '''
  dict_classifier: dizionario del classificatore, con dentro, tra le altre cose, le predizioni del classificatore
  model = la nostra RBM
  test_labels = groundtruth labels
  '''

  Cl_pred_matrix=dict_classifier['Cl_pred_matrix'] #classes predicted by the classifier
  nr_ex=dict_classifier['Cl_pred_matrix'].size()[0] #number of example 

  index = range(model.Num_classes)
  #creo la lista delle categorie delle transizioni ad un certo digit
  to_list = []
  for digit in range(model.Num_classes+1): 
    to_list.append('to_'+ str(digit))

  columns = ['Nr_visited_states','Nr_transitions']+to_list
  #creo i due dataframes: uno per le medie e l'altro per i relativi errori
  df_average = pd.DataFrame(index=index, columns=columns)
  df_sem = pd.DataFrame(index=index, columns=columns)


  for digit in range(model.Num_classes): #per ogni digit...
    digit_idx = test_labels==digit #trovo la posizione del digit tra le labels groundtruth...
    Vis_digit = dict_classifier['Cl_pred_matrix'][digit_idx,:] #trovo le predizioni del classificatore in quelle posizioni
    nr_visited_states_list =[] #qui listerò gli stati che vado a visitare
    nr_transitions_list =[] #qui listerò invece il numero di transizioni che farò
    to_digits_mat = torch.zeros(Vis_digit.size()[0],model.Num_classes+1) #this is a matrix with nr.rows=nr.examples, nr.cols = 10+1 (nr. digits+no digits category)
    
    for nr_ex,example in enumerate(Vis_digit): # example=tensor of the reconstructions category guessed by the classifier
      no10_example=example[example!=10]
      visited_states = torch.unique(no10_example) # find all the states (digits) visited by the RBM (NOTA:NON TENGO CONTO DEL 10)
      nr_visited_states = len(visited_states)
      transitions,counts = torch.unique_consecutive(no10_example,return_counts=True) #HA SENSO FARE COSì?
      nr_transitions = len(transitions)
      to_digits = torch.zeros(model.Num_classes+1)
      transitions,counts = torch.unique_consecutive(example,return_counts=True)

      for state in visited_states:
        idx_state= transitions == state
        to_digits[state.to(torch.long)] = torch.sum(counts[idx_state])

      nr_visited_states_list.append(nr_visited_states)
      nr_transitions_list.append(nr_transitions)
      to_digits_mat[nr_ex,:] = to_digits
    

    df_average.at[digit,'Nr_visited_states'] = round(sum(nr_visited_states_list)/len(nr_visited_states_list),2)
    df_average.at[digit,'Nr_transitions'] = round(sum(nr_transitions_list)/len(nr_transitions_list),2)
    df_average.at[digit,2:] = torch.round(torch.mean(to_digits_mat,0),decimals=2)

    df_sem.at[digit,'Nr_visited_states'] = round(np.std(nr_visited_states_list)/math.sqrt(len(nr_visited_states_list)),2)
    df_sem.at[digit,'Nr_transitions'] = round(np.std(nr_transitions_list)/math.sqrt(len(nr_transitions_list)),2)
    df_sem.at[digit,2:] = torch.round(torch.std(to_digits_mat,0)/math.sqrt(to_digits_mat.size()[0]),decimals=2)
    
    #ratio tra passaggi alla classe giusta e la seconda classe di più alta frequenza
    to_mat = df_average.iloc[:, 2:-1]
    ratio_list =[]
    for index, row in to_mat.iterrows():
        row = row.to_numpy()
        to_2nd_largest = np.amax(np.delete(row, index, None))
        ratio_2nd_on_trueClass = to_2nd_largest/row[index]
        ratio_list.append(ratio_2nd_on_trueClass)

    df_average['Ratio_2nd_trueClass'] = ratio_list

  if Plot==1:
        df_average.plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:, ['Nr_visited_states', 'Nr_transitions']],figsize=(20,10),fontsize=dS)
        plt.title("Classification_metrics-1",fontsize=dS)
        plt.xlabel("Digit",fontsize=dS)
        plt.ylabel("Nr of states",fontsize=dS)
        plt.ylim([0,10])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS)

        df_average.plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(20,10),fontsize=dS,width=0.8,colormap='hsv')
        plt.title("Classification_metrics-2",fontsize=dS)
        plt.xlabel("Digit",fontsize=dS)
        plt.ylabel("Average number of steps",fontsize=dS)
        plt.ylim([0,100])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS)

  return df_average, df_sem

  
  