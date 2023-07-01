# Qui sono inserite tutte le funzioni di plotting utilizzate su colab per ricostruzione immagini
from operator import index
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pandas as pd
import math
import numpy as np
import torch
import random
import itertools
import scipy
import seaborn as sns
from google.colab import files

from Classifiers import Classifier_accuracy
from Classifiers import classification_metrics

def Digitwise_metrics_plot(model,sample_test_labels, sample_test_data,gen_data_dictionary=[], metric_type='cos', dS = 50, l_sz = 5, new_generated_data=False, generated_data=[], temperature=1):
    '''
    metric_type= cos (cosine similarity), energy, perc_act_H (% of activated hidden), actnorm (activation norm(L2) on hid states or probs)
    '''
    
    c=0 #inizializzo il counter per cambiamento colore
    cmap = cm.get_cmap('hsv') # inizializzo la colormap che utilizzerò per il plotting
    figure, axis = plt.subplots(1, 1, figsize=(15,15)) #setto le dimensioni della figura
    lbls = [] # qui storo le labels x legenda
    if new_generated_data:
       result_dict = model.reconstruct(sample_test_data, nr_steps=100, temperature=temperature, include_energy = 1)
    
    for digit in range(model.Num_classes): # per ogni digit...
        
        Color = cmap(c/256) #setto il colore di quel determinato digit
        l = torch.where(sample_test_labels == digit) #trovo gli indici dei test data che contengono quel determinato digit
        nr_examples= len(l[0]) #nr degli esempi di quel digit (i.e. n)

        if metric_type=='cos':
            original_data = sample_test_data[l[0],:,:] #estraggo i dati originali cui confrontare le ricostruzioni
            generated_data = gen_data_dictionary['vis_states'][l[0],:,:] #estraggo le ricostruzioni
            model.cosine_similarity(original_data, generated_data, Plot=1, Color = Color, Linewidth=l_sz) #calcolo la cosine similarity tra 
            #original e generated data
            if digit==0: #evito di fare sta operazione più volte
             y_lbl = 'Cosine similarity'

        elif metric_type=='energy':
            energy_mat_digit = gen_data_dictionary['Energy_matrix'][l[0],:] #mi trovo le entrate della energy matrix relative agli esempi di quel digit
            nr_steps = energy_mat_digit.size()[1] #calcolo il numero di step di ricostruzione a partire dalla energy mat
            SEM = torch.std(energy_mat_digit,0)/math.sqrt(nr_examples) # mi calcolo la SEM
            MEAN = torch.mean(energy_mat_digit,0).cpu() # e la media between examples

            if digit==0: #evito di fare sta operazione più volte
             y_lbl = 'Energy'

        elif metric_type=='actnorm':
            gen_H_digit = gen_data_dictionary['hid_prob'][l[0],:,:] 
            act_norm = gen_H_digit.pow(2).sum(dim=1).sqrt()
            nr_steps = gen_H_digit.size()[2]
            SEM = torch.std(act_norm,0)/math.sqrt(nr_examples)
            MEAN = torch.mean(act_norm,0).cpu()
            if digit==0: #evito di fare sta operazione più volte
             y_lbl = 'Activation (L2) norm'

        else: #perc_act_H


            gen_H_digit = gen_data_dictionary['hid_states'][l[0],:,:]
            nr_steps = gen_H_digit.size()[2]
            if digit == 0:
                Mean_storing = torch.zeros(model.Num_classes,nr_steps, device = 'cuda')
                Sem_storing = torch.zeros(model.Num_classes,nr_steps, device = 'cuda')
            SEM = torch.std(torch.mean(gen_H_digit,1)*100,0)/math.sqrt(nr_examples)
            MEAN = torch.mean(torch.mean(gen_H_digit,1)*100,0).cpu()
            Mean_storing[digit, : ] = MEAN.cuda()
            Sem_storing[digit, : ] = SEM

            if digit==0: #evito di fare sta operazione più volte
             y_lbl = '% active H units'

        if not(metric_type=='cos'):
            SEM = SEM.cpu() #sposto la SEM su CPU x plotting
            x = range(1,nr_steps+1) #asse delle x, rappresentante il nr di step di ricostruzione svolti
            plt.plot(x, MEAN, c = Color, linewidth=l_sz) #plotto la media
            plt.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color, alpha=0.3) # e le barre di errore
        
        c = c+25
        lbls.append(digit)

    axis.legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) # legenda
    #ridimensiono etichette assi e setto le labels
    axis.tick_params(axis='x', labelsize= dS) 
    axis.tick_params(axis='y', labelsize= dS)
    axis.set_ylabel(y_lbl,fontsize=dS)
    axis.set_xlabel('Generation step',fontsize=dS)
    axis.set_title(y_lbl+' - digitwise',fontsize=dS)
    if metric_type=='cos':
      axis.set_ylim([0,1])
    elif metric_type=='perc_act_H':
      axis.set_ylim([0,100])
    #DA FARE SETTARE LIMITI ASSE Y
    if metric_type=='perc_act_H':
      return Mean_storing, Sem_storing


def Average_metrics_plot(model,gen_data_dictionary=[], Intersection_analysis = [],sample_test_data = [], metric_type='cos', dS = 50, l_sz = 5, new_generated_data=False,temperature=1, single_line_plot=True):
  if single_line_plot:
     figure, axis = plt.subplots(1, 1, figsize=(15,15))
     C_list=['blue','lime','black']
  else:
    cmap = cm.get_cmap('hsv')
    cmap(temperature*10/256)
    C_list=[cmap((temperature*15+7*25)/256),cmap((temperature*15+2*25)/256),cmap(temperature*15/256)]

  if new_generated_data:
    if Intersection_analysis == []:
      result_dict = model.reconstruct(sample_test_data, nr_steps=100, temperature=temperature, include_energy = 1)
    else:
      result_dict, df_average = Intersection_analysis.generate_chimera_lbl_biasing(elements_of_interest = [1,7], nr_of_examples = 1000, temperature = temperature)
  
  if metric_type=='cos':
    if new_generated_data:
        MEAN, SEM = model.cosine_similarity(sample_test_data, result_dict['vis_states'], Plot=1, Color = C_list[0],Linewidth=l_sz)
    else:
       #model.cosine_similarity(sample_test_data, model.TEST_vis_states, Plot=1, Color = C_list[0],Linewidth=l_sz) #old code
       MEAN, SEM = model.cosine_similarity(sample_test_data, gen_data_dictionary['vis_states'], Plot=1, Color = C_list[0],Linewidth=l_sz)

    y_lbl = 'Cosine similarity'
  elif metric_type=='energy':
    Color = C_list[1]
    if new_generated_data:
        energy_mat_digit = result_dict['Energy_matrix']
    else:
        #energy_mat_digit = model.TEST_energy_matrix #old
        energy_mat_digit = gen_data_dictionary['Energy_matrix']
    nr_steps = energy_mat_digit.size()[1]
    SEM = torch.std(energy_mat_digit,0)/math.sqrt(energy_mat_digit.size()[0])
    MEAN = torch.mean(energy_mat_digit,0).cpu()
    y_lbl = 'Energy'
  elif metric_type=='actnorm':
    Color = C_list[2]
    if new_generated_data:
        gen_H = result_dict['hid_prob']
    else:    
        #gen_H = model.TEST_gen_hid_prob #old
        gen_H = gen_data_dictionary['hid_prob']
    nr_steps = gen_H.size()[2]
    act_norm = gen_H.pow(2).sum(dim=1).sqrt()
    MEAN = torch.mean(act_norm,0).cpu()
    SEM = (torch.std(act_norm,0)/math.sqrt(gen_H.size()[0]))
    y_lbl = 'Activation (L2) norm'
  else:
    Color = C_list[2]
    if new_generated_data:
        gen_H = result_dict['hid_states']
    else:    
        #gen_H = model.TEST_gen_hid_states #old
        gen_H = gen_data_dictionary['hid_states']
    nr_steps = gen_H.size()[2]
    MEAN = torch.mean(torch.mean(gen_H,1)*100,0).cpu()
    SEM = (torch.std(torch.mean(gen_H,1)*100,0)/math.sqrt(gen_H.size()[0]))
    y_lbl = '% active H units'


  if not(metric_type=='cos'):
    SEM = SEM.cpu()
    x = range(1,nr_steps+1)
    plt.plot(x, MEAN, c = Color, linewidth=l_sz)
    plt.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color,
                  alpha=0.3)
  if single_line_plot:
     axis.tick_params(axis='x', labelsize= dS)
     axis.tick_params(axis='y', labelsize= dS)
     axis.set_ylabel(y_lbl,fontsize=dS)
     axis.set_xlabel('Nr. of steps',fontsize=dS)
     axis.set_title('Average '+y_lbl,fontsize=dS)
     if metric_type=='cos':
        axis.set_ylim([0,1])
     elif metric_type=='perc_act_H':
        axis.set_ylim([0,100])

     plt.show()


  return MEAN,SEM



def single_digit_classification_plots(reconstructed_imgs, dict_classifier, model,temperature=1,row_step=5,dS = 50,lin_sz = 5):
  img_idx =random.randint(0,reconstructed_imgs.size()[0])
  img_idx = range(img_idx,img_idx+1)
  deh =  Reconstruct_plot(reconstructed_imgs[img_idx,:,:],model, nr_steps=100, temperature= temperature,row_step = row_step, d_type='reconstructed')
  figure, axis = plt.subplots(2, figsize=(15,30))
  

  axis[0].plot(dict_classifier['Cl_pred_matrix'].cpu()[img_idx[0],:], linewidth = lin_sz)

  axis[0].tick_params(axis='x', labelsize= dS)
  axis[0].tick_params(axis='y', labelsize= dS)
  axis[0].set_ylabel('Label classification',fontsize=dS)
  axis[0].set_ylim([0,10])
  axis[0].set_yticks(range(0,11))
  axis[0].set_xlabel('Nr. of steps',fontsize=dS)

  axis[1].plot(dict_classifier['Pred_entropy_mat'].cpu()[img_idx[0],:], linewidth = lin_sz, c='r')
  axis[1].tick_params(axis='x', labelsize= dS)
  axis[1].tick_params(axis='y', labelsize= dS)
  axis[1].set_ylabel('Classification entropy',fontsize=dS)
  axis[1].set_ylim([0,2])
  axis[1].set_xlabel('Nr. of steps',fontsize=dS)

  plt.show()


def hidden_states_analysis(d_Reconstruct_t1_allH=[], d_cl=[], Lbl_biasing_probs =[], dS=30, aspect_ratio = 2.5):
  '''
  INPUTS: d_Reconstruct_t1_allH: dictionary obtrained from the reconstruct method of the RBM. It includes visible and hidden states obtained in the generation
  d_cl: dictionary obtained from the classifier accuracy function. It includes the classifications of generated samples
  '''
  tick_labels = ['0','1','2','3','4','5','6','7','8','9','Non\ndigit']

  def single_boxplot(Hid_probs, Color, x_labels = tick_labels):
    df = pd.DataFrame(torch.transpose(Hid_probs,0,1).cpu().numpy())
    distr_percAct_units = sns.catplot(data=df,  kind="box", height=5, aspect=aspect_ratio, palette=Color)
    distr_percAct_units.set_axis_labels("Digit state", "P(h=1)", fontsize=dS)
    _, ylabels = plt.yticks()
    distr_percAct_units.set_yticklabels(ylabels, size=dS)
    #_, xlabels = plt.xticks()

    distr_percAct_units.set_xticklabels(x_labels, size=dS)
    plt.ylim(0, 1)

    #OLD PLOT QUANTIFYiNG avg nr of hidden units active before a certain digit
    # fig, ax = plt.subplots(figsize=(15,10))

    # rects1 = ax.bar(range(11),Active_hid,yerr=Active_hid_SEM, color=Color)
    # ax.set_xlabel('Digit state', fontsize = dS)
    # ax.set_ylabel('Nr of active units', fontsize = dS)
    # ax.set_xticks(range(11))
    # ax.tick_params( labelsize= dS) 
    # ax.set_ylim(0,1000)

  #colori per il plotting
  cmap = cm.get_cmap('hsv') # inizializzo la colormap che utilizzerò per il plotting
  Color = cmap(np.linspace(0, 250, num=11)/256)
  Color[-1]=np.array([0.1, 0.1, 0.1, 1])

  if Lbl_biasing_probs != []:
    Lbl_biasing_probs = torch.transpose(Lbl_biasing_probs,0,1)
    #plot P(h=1) distribution
    single_boxplot(Lbl_biasing_probs, Color, x_labels = [x for x in tick_labels if x != 'Non\ndigit'])

  if d_Reconstruct_t1_allH!=[]:
    average_Hid = torch.zeros(11,1000, device='cuda')
    Active_hid = torch.zeros(11,1, device='cuda')
    Active_hid_SEM = torch.zeros(11,1, device='cuda')
    #for every digit and non-digit class (total:11 classes)...
    for class_of_interest in range(11):
      # Create a tensor of zeros with nrows equal to the number of elements classified as the class of interest, 
      # and 1000 columns (i.e. the number of hidden units of the net)
      Non_num_Hid = torch.zeros(torch.sum(d_cl['Cl_pred_matrix']==class_of_interest),1000)

      counter = 0
      for example in range(1000): #1000 dovrebbe essere il numero totale di campioni generati. Potrebbe essere fatta non hardcoded
        for step in range(100): #nr of generation step
          # Check if the example belongs to the class_of_interest at generation step "step"
          if (d_cl['Cl_pred_matrix']==class_of_interest)[example,step]==True: 
            Non_num_Hid[counter,:]=d_Reconstruct_t1_allH['hid_states'][example,:,step] #insert the corresponding hidden state vector at index "counter"
            counter+=1

      average_Hid[class_of_interest,:]=torch.mean(Non_num_Hid,0) #columnwise average of Non_num_Hid -> P(h=1)
      Active_hid[class_of_interest] = torch.mean(torch.sum(Non_num_Hid,1)) #mean of the sum of elements along the second dimension (axis=1) of a tensor called "Non_num_Hid"
      Active_hid_SEM[class_of_interest] = torch.std(torch.sum(Non_num_Hid,1))/np.sqrt(torch.sum(Non_num_Hid,1).size()[0])
      #print(torch.std(torch.sum(Non_num_Hid,1)),np.sqrt(torch.sum(Non_num_Hid,1).size()[0]) )

    #plot P(h=1) distribution
    single_boxplot(average_Hid, Color)

  
  if Lbl_biasing_probs != [] and d_Reconstruct_t1_allH!=[]:
    pattern = torch.arange(11).repeat_interleave(1000)
    concatenated_tensor = torch.cat((average_Hid.view(-1), Lbl_biasing_probs.reshape(-1)), dim=0)
    concatenated_labels = torch.cat((pattern, pattern[:10000]), dim=0)
    type_vec = ['dataset'] * 11000+['label biasing']*10000

    data_dict = {'P(h=1)': concatenated_tensor.cpu(), 'Digit state': concatenated_labels.cpu(), 'tipo': type_vec}
    dat_f = pd.DataFrame(data_dict)
    
    Color = np.repeat(Color, 2, axis=0)  # repeat each element twice along the first axis
    Color[1::2, 3] = 0.4

    Color=Color[:-1,:]
    fig, ax = plt.subplots(figsize=(5*aspect_ratio, 5))
    sns.boxplot(x='Digit state', y='P(h=1)', hue='tipo',
                 data=dat_f)

    import matplotlib.patches
    boxes = ax.findobj(matplotlib.patches.PathPatch)
    for color, box in zip(Color, boxes):
        box.set_facecolor(color)
    ax.legend_.remove()
    ax.tick_params(labelsize= 30) 
    ax.set_xticklabels(tick_labels, size=dS)
    ax.set_ylabel('P(h=1)',fontsize = dS)
    ax.set_xlabel('Digit state',fontsize = dS)
    ax.set_ylim(0,1)
    plt.plot()

  return average_Hid, Active_hid, Active_hid_SEM



def plot_intersect_count(df_digit_digit_common_elements_count_biasing):

  fig, ax = plt.subplots(figsize=(10,10))

  # hide axes
  fig.patch.set_visible(False)
  ax.axis('off')
  plt.axis(on=None)
  ax.axis('tight')
  rcolors = plt.cm.BuPu(np.full(len(df_digit_digit_common_elements_count_biasing.columns), 0.1))
  colV = []
  for digit in range(10):
    colV.append('Digit: '+str(digit))
  table = ax.table(cellText=df_digit_digit_common_elements_count_biasing.values, colLabels=colV,
          rowLabels=colV, rowColours=rcolors, rowLoc='right',
                        colColours=rcolors, loc='center', fontsize=20)

  fig.tight_layout()

  from matplotlib.font_manager import FontProperties

  for (row, col), cell in table.get_celld().items():
    if (row == 0) or (col == -1):
      cell.set_text_props(fontproperties=FontProperties(weight='bold'))


def top_k_generation(VGG_cl, model,n_rep=100, nr_steps=100, temperature=1, k=100, entropy_correction=1):
  vis_lbl_bias, gen_hidden_act=model.label_biasing(nr_steps=nr_steps)
  #processing of lbl biasing vec for reconstruct from hidden
  gen_hidden_act = torch.transpose(gen_hidden_act, 0,1)
  gen_hidden_act = gen_hidden_act.repeat(100, 1)
  gen_hidden_act = torch.unsqueeze(gen_hidden_act, 2)

  #do the reconstruction from label biasing vector with k units active
  d = model.reconstruct_from_hidden(gen_hidden_act , nr_steps=nr_steps, temperature=temperature, include_energy = 1,consider_top=k)

  #compute classifier accuracy and entropy
  LblBiasGenerated_imgs=d['vis_states']
  VStack_labels=torch.tensor(range(10), device = 'cuda')
  VStack_labels=VStack_labels.repeat(100)
  d_cl = Classifier_accuracy(LblBiasGenerated_imgs, VGG_cl, model, labels=VStack_labels, entropy_correction=entropy_correction, plot=0)

  return d_cl['Cl_accuracy'][-1],d_cl['MEAN_entropy'][-1], d_cl['digitwise_acc'][:,-1]

def Cl_plot(axis,x,y,y_err=[],x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=3, dS=30, color='g'):
  y=y.cpu()
  
  axis.plot(x, y, c = color, linewidth=l_sz)
  if y_err != []:
    y_err = y_err.cpu()
    axis.fill_between(x,y-y_err, y+y_err, color=color,
                alpha=0.3)
  axis.tick_params(axis='x', labelsize= dS)
  axis.tick_params(axis='y', labelsize= dS)
  axis.set_ylabel(y_lab,fontsize=dS)
  axis.set_ylim(lim_y)
  axis.set_xlabel(x_lab,fontsize=dS)
  axis.set_title(Title,fontsize=dS)
