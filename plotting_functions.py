# Qui sono inserite tutte le funzioni di plotting utilizzate su colab per ricostruzione immagini
from operator import index
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np
import torch
import random

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
  axis.set_xlabel('Nr. reconstruction steps',fontsize=dS)
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


def Digitwise_metrics_plot(model, sample_test_data, metric_type='cos', dS = 50, l_sz = 5, new_generated_data=False, generated_data=[], temperature=1):
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
        l = torch.where(model.TEST_lbls == digit) #trovo gli indici dei test data che contengono quel determinato digit
        nr_examples= len(l[0]) #nr degli esempi di quel digit (i.e. n)

        if metric_type=='cos':
            original_data = sample_test_data[l[0],:,:] #estraggo i dati originali cui confrontare le ricostruzioni
            generated_data = model.TEST_vis_states[l[0],:,:] #estraggo le ricostruzioni
            model.cosine_similarity(original_data, generated_data, Plot=1, Color = Color, Linewidth=l_sz) #calcolo la cosine similarity tra 
            #original e generated data
            if digit==0: #evito di fare sta operazione più volte
             y_lbl = 'Cosine similarity'

        elif metric_type=='energy':
            energy_mat_digit = model.TEST_energy_matrix[l[0],:] #mi trovo le entrate della energy matrix relative agli esempi di quel digit
            nr_steps = energy_mat_digit.size()[1] #calcolo il numero di step di ricostruzione a partire dalla energy mat
            SEM = torch.std(energy_mat_digit,0)/math.sqrt(nr_examples) # mi calcolo la SEM
            MEAN = torch.mean(energy_mat_digit,0).cpu() # e la media between examples

            if digit==0: #evito di fare sta operazione più volte
             y_lbl = 'Energy'

        elif metric_type=='actnorm':
            gen_H_digit = model.TEST_gen_hid_prob[l[0],:,:]
            act_norm = gen_H_digit.pow(2).sum(dim=1).sqrt()
            nr_steps = gen_H_digit.size()[2]
            SEM = torch.std(act_norm,0)/math.sqrt(nr_examples)
            MEAN = torch.mean(act_norm,0).cpu()
            if digit==0: #evito di fare sta operazione più volte
             y_lbl = 'Activation (L2) norm'

        else: #perc_act_H
            gen_H_digit = model.TEST_gen_hid_states[l[0],:,:]
            nr_steps = gen_H_digit.size()[2]
            SEM = torch.std(torch.mean(gen_H_digit,1)*100,0)/math.sqrt(nr_examples)
            MEAN = torch.mean(torch.mean(gen_H_digit,1)*100,0).cpu()
            
            if digit==0: #evito di fare sta operazione più volte
             y_lbl = '% active h units'

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
    axis.set_xlabel('Nr. reconstruction steps',fontsize=dS)
    axis.set_title(y_lbl+' - digitwise',fontsize=dS)
    #DA FARE SETTARE LIMITI ASSE Y


def Average_metrics_plot(model, sample_test_data, metric_type='cos', dS = 50, l_sz = 5, new_generated_data=False,temperature=1, single_line_plot=True):
  if single_line_plot:
     figure, axis = plt.subplots(1, 1, figsize=(15,15))
     C_list=['blue','lime','black']
  else:
    cmap = cm.get_cmap('hsv')
    cmap(temperature*10/256)
    C_list=[cmap((temperature*15+7*25)/256),cmap((temperature*15+2*25)/256),cmap(temperature*15/256)]

  if new_generated_data:
     result_dict = model.reconstruct(sample_test_data, nr_steps=100, temperature=temperature, include_energy = 1)
  
  if metric_type=='cos':
    if new_generated_data:
        model.cosine_similarity(sample_test_data, result_dict['vis_states'], Plot=1, Color = C_list[0],Linewidth=l_sz)
    else:
       model.cosine_similarity(sample_test_data, model.TEST_vis_states, Plot=1, Color = C_list[0],Linewidth=l_sz)
    y_lbl = 'Cosine similarity'
  elif metric_type=='energy':
    Color = C_list[1]
    if new_generated_data:
        energy_mat_digit = result_dict['Energy_matrix']
    else:
        energy_mat_digit = model.TEST_energy_matrix
    nr_steps = energy_mat_digit.size()[1]
    SEM = torch.std(energy_mat_digit,0)/math.sqrt(energy_mat_digit.size()[0])
    MEAN = torch.mean(energy_mat_digit,0).cpu()
    y_lbl = 'Energy'
  elif metric_type=='actnorm':
    Color = C_list[2]
    if new_generated_data:
        gen_H = result_dict['hid_prob']
    else:    
        gen_H = model.TEST_gen_hid_prob
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
        gen_H = model.TEST_gen_hid_states 
    nr_steps = gen_H.size()[2]
    MEAN = torch.mean(torch.mean(gen_H,1)*100,0).cpu()
    SEM = (torch.std(torch.mean(gen_H,1)*100,0)/math.sqrt(gen_H.size()[0]))
    y_lbl = '% active h units'

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
     axis.set_xlabel('Nr. reconstruction steps',fontsize=dS)
     axis.set_title('Average '+y_lbl,fontsize=dS)
     if metric_type=='cos':
        axis.set_ylim([0,1])
     plt.show()
  


def Cosine_hidden_plot(model,  dS = 20, l_sz = 5):
  S1_pHid = model.TEST_gen_hid_prob[:,:,0]
  cmap = cm.get_cmap('hsv')
  figure, axis = plt.subplots(1, model.Num_classes, figsize=(50,5))
  lbls = range(model.Num_classes)
  ref_mat = torch.zeros([model.Num_classes,1000], device =model.DEVICE)

  for digit in range(model.Num_classes):
      
      l = torch.where(model.TEST_lbls == digit)
      Hpr_digit = S1_pHid[l[0],:]
      ref_mat[digit,:] = torch.mean(Hpr_digit,0)
  
  for digit_plot in range(model.Num_classes):
      c=0
      l = torch.where(model.TEST_lbls == digit_plot)
      Hpr_digit = model.TEST_gen_hid_prob[l[0],:,:]
      for digit in range(model.Num_classes):
          model.cosine_similarity(ref_mat[digit:digit+1,:], Hpr_digit, Plot=1, Color = cmap(c/256), Linewidth=l_sz, axis=axis[digit_plot])
          c = c+25
      if digit_plot==9:
        axis[digit_plot].legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #cambia posizione
      axis[digit_plot].tick_params(axis='x', labelsize= dS)
      axis[digit_plot].tick_params(axis='y', labelsize= dS)
      axis[digit_plot].set_ylabel('Cosine similarity',fontsize=dS)
      axis[digit_plot].set_ylim([0,1])
      axis[digit_plot].set_xlabel('Nr. reconstruction steps',fontsize=dS)
      axis[digit_plot].set_title("Digit: {}".format(digit_plot),fontsize=dS)  

        #da finire 05 07
  plt.subplots_adjust(left=0.1, 
                      bottom=0.1,  
                      right=0.9,  
                      top=0.9,  
                      wspace=0.4,  
                      hspace=0) 

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
  axis[0].set_xlabel('Nr. reconstruction steps',fontsize=dS)

  axis[1].plot(dict_classifier['Pred_entropy_mat'][img_idx[0],:], linewidth = lin_sz, c='r')
  axis[1].tick_params(axis='x', labelsize= dS)
  axis[1].tick_params(axis='y', labelsize= dS)
  axis[1].set_ylabel('Classification entropy',fontsize=dS)
  axis[1].set_ylim([0,2])
  axis[1].set_xlabel('Nr. reconstruction steps',fontsize=dS)

  plt.show()










