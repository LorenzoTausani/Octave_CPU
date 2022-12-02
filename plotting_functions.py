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

from VGG_MNIST import Classifier_accuracy
from VGG_MNIST import classification_metrics

def mean_h_prior(model):
  mean_h_prob_mat = torch.zeros(model.Num_classes+1,model.layersize[0]).to(model.DEVICE)
  gen_H = model.TRAIN_gen_hid_prob[:,:,0]

  for it in range(model.Num_classes+1):
    if it>9:
      mean_h_prob_mat[it,:] = torch.mean(gen_H,0)
    else:
      l = torch.where(model.TRAIN_lbls == it)
      gen_H_digit = gen_H[l[0],:]
      mean_h_prob_mat[it,:] = torch.mean(gen_H_digit,0)

  mean_h_prob_mat=torch.unsqueeze(mean_h_prob_mat,2)
  return mean_h_prob_mat



def Between_model_Cl_accuracy(models_list, nr_steps, dS = 50, l_sz = 5):
  #questa funzione plotta l'accuratezza dei classificatori lineari sugli hidden states al variare del nr di steps di ricostruzione
  # between diversi modelli RBM

  figure, axis = plt.subplots(1, 1, figsize=(15,15)) #costruisco la figura che conterrà il plot

  lbls = [] #qui andrò a storare il nr di epoche per cui sono stati trainati i modelli, per poi utilizzarlo nella legenda
  x = range(1,nr_steps+1) #questo vettore, che stora il nr di steps, è usato per il plotting

  c=30 #questo counter è usato per determinare il colore della linea alla iterazione corrente
  cmap = cm.get_cmap('hsv') #utilizzo questa colormap
  for model in models_list: #per ogni modello loaddato...
    axis.plot(x,model.Cl_TEST_step_accuracy[:nr_steps], linewidth=l_sz, markersize=12,marker='o', c=cmap(c/256))
    c = c+30 #cambio colore per il plossimo line plot
    lbls.append(model.maxepochs) #il nr di epoche per cui è stato trainato il modello lo storo qui, per poi utilizzarlo nella legenda 
  #qua sotto cambio il size di scrittura dei ticks sui due assi
  axis.tick_params(axis='x', labelsize= dS) 
  axis.tick_params(axis='y', labelsize= dS)

  axis.legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #imposto la legenda
  #setto i nomi degli assi e il titolo del plot
  axis.set_ylabel('Linear classifier accuracy',fontsize=dS)
  axis.set_xlabel('Nr. of steps',fontsize=dS)
  axis.set_title('Classifier accuracy',fontsize=dS)

  #axis.set_xticks(np.arange(0, nr_steps+1, 1))
  axis.set_yticks(np.arange(0, 1, 0.1)) #setto il range di y tra 0 e 1 (che è la max accuracy)
  plt.show()


def Reconstruct_plot(input_data, model, nr_steps=100, temperature= 1,row_step = 10, d_type='example', consider_top = 1000):
    '''
    INPUT: 
    input_data: possono essere o dataset da ricostruire (in tal caso d_type='example'), o visible ottenuti da label biasing (in tal caso d_type='lbl_biasing')
    o dati già ricostruiti (in tal caso d_type='reconstructed'), o un input hidden unit activation (in tal caso d_type='hidden')
    '''

    rows = math.floor(nr_steps/row_step) #calcolo il numero di rows che dovrà avere il mio plot
    
    #calcolo il numero di colonne che dovrà avere il mio plot, e in funzione di quello imposto la figsize
    if not(d_type=='example' or d_type=='lbl_biasing'): 
        cols = input_data.size()[0] 
        figure, axis = plt.subplots(rows+1,cols, figsize=(25*(cols/10),2.5*(1+rows))) 
        if cols==1:
            axis= np.expand_dims(axis, axis=1) #aggiungo una dimensione=1 così che non ho problemi nell'indicizzazione di axis
        if d_type=='hidden':
            d= model.reconstruct_from_hidden(input_data , nr_steps, temperature=temperature, consider_top=consider_top) #faccio la ricostruzione da hidden
            input_data=d['vis_states'] #estraggo le immagini ricostruite


    else: # nel caso di reconstruct examples o label biasing
        cols = model.Num_classes #le colonne sono 10 in quanto 10 sono i digits
        figure, axis = plt.subplots(rows+2, cols, figsize=(25,2.5*(2+rows))) # 2 sta per originale+ 1 step reconstruction, che ci sono sempre 
        good_digits_idx = [71,5,82,32,56,15,21,64,110,58] #bei digits selezionati manualmente da me (per example)
        if d_type=='example':
          orig_data = input_data # copio in questo modo per poi ottenere agilmente i dati originali
          d= model.reconstruct(input_data.data[good_digits_idx].to(model.DEVICE),nr_steps, temperature=temperature, consider_top=consider_top) #faccio la ricostruzione
        else:
          orig_data = input_data.view((10,28,28))
          d= model.reconstruct(orig_data,nr_steps, temperature=temperature, consider_top=consider_top) #faccio la ricostruzione
        input_data=d['vis_states'] #estraggo le immagini ricostruite
    
    for lbl in range(cols): #per ogni digit...
        
        if  d_type=='example' or d_type=='lbl_biasing':
            before = 1 # perchè c'è anche il plot dell'originale/biasing
            # plotto l'originale (i.e. non ricostruito)
            if d_type=='example': #differenzio tra example e biasing perchè diverso è il tipo di dato in input
              axis[0, lbl].imshow(orig_data.data[good_digits_idx[lbl]] , cmap = 'gray')
              axis[0, lbl].set_title("Original number:{}".format(lbl))
            else:
              axis[0, lbl].imshow(orig_data[lbl,:,:].cpu() , cmap = 'gray')
              axis[0, lbl].set_title("lbl biasing number:{}".format(lbl))
            axis[0, lbl].set_xticklabels([])
            axis[0, lbl].set_yticklabels([])
            axis[0, lbl].set_aspect('equal')

        else:
            before = 0 # non ho il plot dell'originale

        # plotto la ricostruzione dopo uno step
        reconstructed_img= input_data[lbl,:,0] #estraggo la prima immagine ricostruita per il particolare esempio (lbl può essere un nome un po fuorviante)
        reconstructed_img = reconstructed_img.view((28,28)).cpu() #ridimensiono l'immagine e muovo su CPU

        axis[before, lbl].imshow(reconstructed_img , cmap = 'gray')
        axis[before, lbl].set_title("Rec.-step {}".format(1))
        axis[before, lbl].set_xticklabels([])
        axis[before, lbl].set_yticklabels([])
        axis[before, lbl].set_aspect('equal')
 
        for idx,step in enumerate(range(row_step,nr_steps+1,row_step)): # idx = riga dove plotterò, step è il recostruction step che ci plotto
            idx = idx+before+1 #sempre +1 perchè c'è sempre 1 step reconstruction (+1 se before=1 perchè c'è anche l'originale)
            
            #plotto la ricostruzione
            reconstructed_img= input_data[lbl,:,step-1] #step-1 perchè 0 è la prima ricostruzione
            reconstructed_img = reconstructed_img.view((28,28)).cpu()
            axis[idx, lbl].imshow(reconstructed_img , cmap = 'gray')
            axis[idx, lbl].set_title("Rec.-step {}".format(step))
            axis[idx, lbl].set_xticklabels([])
            axis[idx, lbl].set_yticklabels([])
            axis[idx, lbl].set_aspect('equal')
    
    #aggiusto gli spazi tra le immagini
    plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.2) 
    
    #plt.savefig("Reconstuct_plot.jpg") #il salvataggio è disabilitato

    plt.show()

    if not(d_type=='reconstructed'): #nel caso in cui si siano operate ricostruzioni
      return d #restituisci l'output della ricostruzione


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
            SEM = torch.std(torch.mean(gen_H_digit,1)*100,0)/math.sqrt(nr_examples)
            MEAN = torch.mean(torch.mean(gen_H_digit,1)*100,0).cpu()
            
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
    axis.set_xlabel('Nr. of steps',fontsize=dS)
    axis.set_title(y_lbl+' - digitwise',fontsize=dS)
    if metric_type=='cos':
      axis.set_ylim([0,1])
    elif metric_type=='perc_act_H':
      axis.set_ylim([0,100])
    #DA FARE SETTARE LIMITI ASSE Y


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
  
  
  


def Cosine_hidden_plot(model,gen_data_dictionary, sample_test_labels, dS = 40, l_sz = 5):
  #S1_pHid = model.TEST_gen_hid_prob[:,:,0]
  S1_pHid = gen_data_dictionary['hid_prob'][:,:,0]
  cmap = cm.get_cmap('hsv')
  figure, axis = plt.subplots(1, model.Num_classes, figsize=(50,10))
  lbls = range(model.Num_classes)
  ref_mat = torch.zeros([model.Num_classes,1000], device =model.DEVICE)
  MEAN_cosSim_tensor=torch.empty((model.Num_classes,gen_data_dictionary['hid_prob'].size()[2],model.Num_classes))
  SEM_cosSim_tensor=torch.empty((model.Num_classes,gen_data_dictionary['hid_prob'].size()[2],model.Num_classes))

  for digit in range(model.Num_classes):
      
      l = torch.where(sample_test_labels == digit)
      Hpr_digit = S1_pHid[l[0],:]
      ref_mat[digit,:] = torch.mean(Hpr_digit,0)
  
  for digit_plot in range(model.Num_classes):
      c=0
      l = torch.where(sample_test_labels == digit_plot)
      Hpr_digit = gen_data_dictionary['hid_prob'][l[0],:,:]
      for digit in range(model.Num_classes):
          MEAN, SEM = model.cosine_similarity(ref_mat[digit:digit+1,:], Hpr_digit, Plot=1, Color = cmap(c/256), Linewidth=l_sz, axis=axis[digit_plot])
          if digit==0:
            digit_plot_mat_MEAN = MEAN
            digit_plot_mat_SEM = SEM
          else:
            digit_plot_mat_MEAN = torch.vstack((digit_plot_mat_MEAN,MEAN))
            digit_plot_mat_SEM = torch.vstack((digit_plot_mat_SEM,SEM))
          c = c+25
      #print(digit_plot_mat_MEAN)
      MEAN_cosSim_tensor[:,:,digit_plot] = digit_plot_mat_MEAN
      SEM_cosSim_tensor[:,:,digit_plot] = digit_plot_mat_SEM

      if digit_plot==9:
        axis[digit_plot].legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #cambia posizione
      axis[digit_plot].tick_params(axis='x', labelsize= dS)
      axis[digit_plot].tick_params(axis='y', labelsize= dS)
      if digit_plot==0:
        axis[digit_plot].set_ylabel('Cosine similarity',fontsize=dS)
      else:
        axis[digit_plot].set_yticklabels([])
      axis[digit_plot].set_ylim([0,1])
      axis[digit_plot].set_xlabel('Nr. of steps',fontsize=dS)
      axis[digit_plot].set_title("Digit: {}".format(digit_plot),fontsize=dS)  

        #da finire 05 07
  plt.subplots_adjust(left=0.1, 
                      bottom=0.1,  
                      right=0.9,  
                      top=0.9,  
                      wspace=0.15,  
                      hspace=0) 
  return MEAN_cosSim_tensor, SEM_cosSim_tensor


def single_digit_classification_plots(reconstructed_imgs, dict_classifier, model,temperature=1,row_step=5,dS = 50,lin_sz = 5):
  img_idx =random.randint(0,reconstructed_imgs.size()[0])
  img_idx = range(img_idx,img_idx+1)
  deh =  Reconstruct_plot(reconstructed_imgs[img_idx,:,:],model, nr_steps=100, temperature= temperature,row_step = row_step, d_type='reconstructed')
  figure, axis = plt.subplots(2, figsize=(15,30))
  

  axis[0].plot(dict_classifier['Cl_pred_matrix'][img_idx[0],:], linewidth = lin_sz)

  axis[0].tick_params(axis='x', labelsize= dS)
  axis[0].tick_params(axis='y', labelsize= dS)
  axis[0].set_ylabel('Label classification',fontsize=dS)
  axis[0].set_ylim([0,10])
  axis[0].set_yticks(range(0,11))
  axis[0].set_xlabel('Nr. of steps',fontsize=dS)

  axis[1].plot(dict_classifier['Pred_entropy_mat'][img_idx[0],:], linewidth = lin_sz, c='r')
  axis[1].tick_params(axis='x', labelsize= dS)
  axis[1].tick_params(axis='y', labelsize= dS)
  axis[1].set_ylabel('Classification entropy',fontsize=dS)
  axis[1].set_ylim([0,2])
  axis[1].set_xlabel('Nr. of steps',fontsize=dS)

  plt.show()

def between_VH_plots(dict_VH_LayerState_comparison,dS=23,yLab = 'Accuracy'):
  #pairwise stat tests
  comparisons=list(itertools.combinations(dict_VH_LayerState_comparison.keys(),2))
  significant_comparisons=[]

  for comparison in comparisons:
    campione1=np.asarray(dict_VH_LayerState_comparison[comparison[0]])
    campione2=np.asarray(dict_VH_LayerState_comparison[comparison[1]])
    U, p = scipy.stats.wilcoxon(campione1,campione2, alternative='two-sided')
    if p*len(comparisons) < 0.05: #bonferroni correction
        significant_comparisons.append([comparison, p*len(comparisons)])


  cmap = cm.get_cmap('hsv')
  cat=0
  x_labs = []
  plt.figure(figsize=(8,8))
  for VH_state in dict_VH_LayerState_comparison:
    c=0
    for y in dict_VH_LayerState_comparison[VH_state]:
          
        # plotting the corresponding x with y 
        # and respective color
        plt.scatter(cat, y, c = cmap(c/256), s = 50, linewidth = 0)
        c+=25
    cat+=1
    if VH_state[0]=='continous':
      xLab='Vc_'
    else:
      xLab='Vb_'
    if VH_state[1]=='continous':
      xLab=xLab+'Hc'
    else:
      xLab=xLab+'Hb'

    x_labs.append(xLab)
  
  if yLab == 'Accuracy':
    bottom=0
    top=1
  elif yLab == 'Nr of transitions':
    bottom=0
    top=20
  elif yLab == 'Nr steps in no digit state':
    bottom=0
    top=50
  else: #nr of visible states
    bottom=0
    top=5    
    

  plt.ylabel(yLab, fontsize = dS)
  plt.tick_params('both',labelsize=dS)
  plt.xlabel('Visible and hiddel layer type', fontsize = dS)      
  plt.xticks(range(3), x_labs)
  plt.yticks(np.arange (bottom, top+top/4, top/4))
  plt.ylim(bottom,top+top/4)
  
  y_range = top - bottom
  for i, significant_combination in enumerate(significant_comparisons):
      # Columns corresponding to the datasets of interest

      x1 = 0
      for comb in significant_combination[0][0]:
        if comb=='binary':
          x1 +=1
      x2 = 0
      for comb in significant_combination[0][1]:
        if comb=='binary':
          x2 +=1

      # What level is this bar among the bars above the plot?
      level = len(significant_comparisons) - i
      # Plot the bar
      bar_height = (y_range * 0.07 * level) + top
      bar_tips = bar_height - (y_range * 0.02)
      plt.plot([x1, x1, x2, x2],[bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
      # Significance level
      p = significant_combination[1]
      if p < 0.001:
          sig_symbol = '***'
      elif p < 0.01:
          sig_symbol = '**'
      elif p < 0.05:
          sig_symbol = '*'
      text_height = bar_height + (y_range * 0.01)
      plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', fontsize=13)


  plt.show()


def Comparison_VH_LayerState(sample_test_data,sample_test_labels, nr_steps, temperature,consider_top_H, VGG_cl, model, plot=1):
  '''
  #Se si vuole avere anche VbHc

  LayerState_type=['continous','binary']
  VH_LayerStates = [ls for ls in itertools.product(LayerState_type, repeat=2)]
  '''
  VH_LayerStates=list(itertools.combinations_with_replacement(['continous','binary'],2))
  dict_VHstate_accuracy={}
  dict_VHstate_nrVisitedSts = {}
  dict_VHstate_nrTransitions={}
  dict_VHstate_toNoNum={}

  for VH_state in VH_LayerStates:
    model.Visible_mode = VH_state[0]
    model.Hidden_mode = VH_state[1]

    d= model.reconstruct(sample_test_data, nr_steps, temperature=temperature,consider_top=consider_top_H)
    reconstructed_imgs=d['vis_states']
    d_cl = Classifier_accuracy(reconstructed_imgs, VGG_cl, model, labels=sample_test_labels, plot=0)
    df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d_cl,model,sample_test_labels, Plot=0)

    acc_digitwise=[]
    for idx in range(10):
      acc_digitwise.append(d_cl['digitwise_acc'][idx][-1])
    dict_VHstate_accuracy[VH_state]=acc_digitwise
    dict_VHstate_nrVisitedSts[VH_state]=df_average['Nr_visited_states'].to_numpy()
    dict_VHstate_nrTransitions[VH_state]=df_average['Nr_transitions'].to_numpy()
    dict_VHstate_toNoNum[VH_state] =df_average['Non-digit'].to_numpy()

  if plot==1:
    between_VH_plots(dict_VHstate_accuracy)
    between_VH_plots(dict_VHstate_nrVisitedSts,yLab = 'Nr of visited states')
    between_VH_plots(dict_VHstate_nrTransitions,yLab = 'Nr of transitions')
    between_VH_plots(dict_VHstate_toNoNum,yLab = 'Nr steps in no digit state')

  return dict_VHstate_nrTransitions, dict_VHstate_nrVisitedSts, dict_VHstate_toNoNum


def hidden_states_analysis(d_Reconstruct_t1_allH,d_cl, dS=30):
  average_Hid = torch.zeros(11,1000)
  Active_hid = torch.zeros(11,1)
  Active_hid_SEM = torch.zeros(11,1)
  for class_of_interest in range(11):

    Non_num_Hid = torch.zeros(torch.sum(d_cl['Cl_pred_matrix']==class_of_interest),1000)

    counter = 0
    for example in range(1000):
      for step in range(100):
        if (d_cl['Cl_pred_matrix']==class_of_interest)[example,step]==True:
          Non_num_Hid[counter,:]=d_Reconstruct_t1_allH['hid_states'][example,:,step]
          counter+=1

    average_Hid[class_of_interest,:]=torch.mean(Non_num_Hid,0)
    Active_hid[class_of_interest] = torch.mean(torch.sum(Non_num_Hid,1))
    Active_hid_SEM[class_of_interest] = torch.std(torch.sum(Non_num_Hid,1))/np.sqrt(torch.sum(Non_num_Hid,1).size()[0])
    #print(torch.std(torch.sum(Non_num_Hid,1)),np.sqrt(torch.sum(Non_num_Hid,1).size()[0]) )

  #plot P(h=1) distribution
  df = pd.DataFrame(torch.transpose(average_Hid,0,1).numpy())

  distr_percAct_units = sns.catplot(data=df,  kind="box", height=5, aspect=1.5)
  distr_percAct_units.set_axis_labels("Digit state", "P(h=1)", fontsize=dS)
  _, ylabels = plt.yticks()
  distr_percAct_units.set_yticklabels(ylabels, size=dS)
  _, xlabels = plt.xticks()
  distr_percAct_units.set_xticklabels(xlabels, size=dS)
  plt.ylim(0, 1)

  fig, ax = plt.subplots(figsize=(15,10))
  cmap = cm.get_cmap('hsv') # inizializzo la colormap che utilizzerò per il plotting
  Color = cmap(np.linspace(0, 250, num=11)/256)
  Color[-1]=np.array([0.1, 0.1, 0.1, 1])
  rects1 = ax.bar(range(11),Active_hid,yerr=Active_hid_SEM, color=Color)
  ax.set_xlabel('Digit state', fontsize = dS)
  ax.set_ylabel('Nr of active units', fontsize = dS)
  ax.set_xticks(range(11))
  ax.tick_params( labelsize= dS) 
  ax.set_ylim(0,1000)

  return average_Hid, Active_hid, Active_hid_SEM


def between_temperatures_analysis(model, VGG_cl, Ian, sample_test_data, sample_test_labels,type='sample_reconstruct', elements_of_interest = [1,7], t_beginning = 0.1,t_end = 2, t_step=0.1, consider_top_H=1000, plot = 'yes'):
  temperatures = np.arange(t_beginning, t_end, t_step)
  variables_of_interest = ['Nr_visited_states', 'Nr_transitions','Non-digit'] #,'Ratio_2nd_trueClass'
  AxisLabels_dict={'Nr_visited_states':'Nr of visited states', 'Nr_transitions': 'Nr of state transitions', 'Non-digit': 'Non-digit state time'}
  results_mat = np.zeros((len(temperatures),len(variables_of_interest)))  
  err_mat = np.zeros((len(temperatures),len(variables_of_interest)))
  c=0
  for t in temperatures:
    if not(type=='chimera'):
      if type=='sample_reconstruct':
        d= model.reconstruct(sample_test_data, 100, temperature=t, consider_top=consider_top_H)
        reconstructed_imgs=d['vis_states']
        d_cl = Classifier_accuracy(reconstructed_imgs, VGG_cl, model, labels=sample_test_labels, plot=0)
        df_average,df_sem, T_mat = classification_metrics(d_cl,model,sample_test_labels, Plot=0) 

      elif type=='label biasing':
        vis_lbl_bias, gen_hidden_act=model.label_biasing(nr_steps=100)
        for n in range(100): #100 is for 1000 total label biasing digits
          if n==0:
            VStack_lblBias = vis_lbl_bias
          else:
            VStack_lblBias = torch.vstack((VStack_lblBias,vis_lbl_bias))
            
        VStack_lblBias = VStack_lblBias.view((1000,28,28))
        d= model.reconstruct(VStack_lblBias, nr_steps=100, temperature=t,consider_top=consider_top_H) 
        reconstructed_imgs=d['vis_states']
        VStack_labels=torch.tensor(range(10), device = 'cuda')
        VStack_labels=VStack_labels.repeat(100)
        d_cl = Classifier_accuracy(reconstructed_imgs, VGG_cl, model, labels=VStack_labels, plot=0)
        df_average,df_sem,T_mat = classification_metrics(d_cl,model,VStack_labels, Plot=0)

      elif type=='avg hidden biasing':
        Avg_hid_digits = mean_h_prior(model)[:10,:,:]
        for n in range(100): #100 is for 1000 total label biasing digits
          if n==0:
            VStack_avgHid_digits = Avg_hid_digits
          else:
            VStack_avgHid_digits = torch.vstack((VStack_avgHid_digits,Avg_hid_digits))

        d = model.reconstruct_from_hidden(VStack_avgHid_digits , nr_steps=100, temperature=t, consider_top=consider_top_H) #faccio la ricostruzione da hidden
        
        reconstructed_imgs=d['vis_states']
        VStack_labels=torch.tensor(range(10), device = 'cuda')
        VStack_labels=VStack_labels.repeat(100)
        d_cl = Classifier_accuracy(reconstructed_imgs, VGG_cl, model, labels=VStack_labels, plot=0)
        df_average,df_sem,T_mat = classification_metrics(d_cl,model,VStack_labels, Plot=0)


    elif type=='chimera':
      #d, df_average = Ian.generate_chimera_lbl_biasing(VGG_cl, elements_of_interest = elements_of_interest, nr_of_examples = sample_test_labels.size()[0], temperature = t)
      d, df_average, df_sem = Ian.generate_chimera_lbl_biasing(VGG_cl,elements_of_interest = elements_of_interest, nr_of_examples = 1000, temperature = t)
    m = df_average[variables_of_interest].mean()
    var_idx_count = 0
    for var in variables_of_interest:
      err_mat[c,var_idx_count]=error_propagation(df_average[var], df_sem[var], operation = 'average')
      var_idx_count +=1


    results_mat[c,:] = m.to_numpy()

    c+=1

  results_mat = pd.DataFrame(results_mat, columns = variables_of_interest)
  err_mat = pd.DataFrame(err_mat, columns = variables_of_interest)

  #results_mat['temperature'] = temperatures

  if plot == 'yes':
    dS=15
    figure, axis = plt.subplots(1, 3, figsize=(15,5))
    C_list=['blue','red','green','orange']
    counterColor = 0

    for colName, ax in zip(results_mat, axis.ravel()):

      ax.errorbar(temperatures, results_mat[colName], yerr=err_mat[colName],linewidth = 2.5 ,marker='o', c=C_list[counterColor])
      counterColor += 1

      ax.tick_params(axis='x', labelsize= dS)
      ax.tick_params(axis='y', labelsize= dS)
      ax.set_ylabel(AxisLabels_dict[colName],fontsize=dS)
      ax.set_xlabel('Temperature',fontsize=dS)
      if colName=='Nr_visited_states':
        ax.set_ylim([0,10])
      elif colName=='Non-digit':
        ax.set_ylim([0,100])
      else:
        ax.set_ylim([0,20])




  plt.subplots_adjust(left=0.1, 
                      bottom=0.1,  
                      right=0.9,  
                      top=0.9,  
                      wspace=0.3,  
                      hspace=0) 
    #figure.suptitle('Between temperatures generativity',fontsize=dS*3)

  return results_mat, err_mat


def similarity_between_temperatures(model,sample_test_data,temperatures_for_comparisons, metric_type='cos', dS=30):

  figure, axis = plt.subplots(1, 1, figsize=(15,15))
  for t in temperatures_for_comparisons:
    Average_metrics_plot(model, sample_test_data, temperature=t, new_generated_data=True, single_line_plot=False, metric_type=metric_type)
  
  if metric_type=='cos':
    Y_lim = [0,1]
    y_lbl = 'Cosine similarity'
  elif metric_type == 'perc_act_H':
    Y_lim = [0,100]
    y_lbl = '% active H units'
  else:
    Y_lim = [-200,500]
    y_lbl = 'Model energy'

  axis.tick_params(axis='x', labelsize= dS)
  axis.tick_params(axis='y', labelsize= dS)
  axis.set_ylabel(y_lbl,fontsize=dS)
  axis.set_xlabel('Nr. of steps',fontsize=dS)
  axis.set_title('Average '+y_lbl,fontsize=dS)
  axis.set_ylim(Y_lim)
  axis.legend(temperatures_for_comparisons, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) # legenda

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