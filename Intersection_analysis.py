import torch
import random
import VGG_MNIST
import plotting_functions
import methods
from VGG_MNIST import *
from plotting_functions import *
from methods import *
from google.colab import files
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

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

class Intersection_analysis:
    def __init__(self, model, top_k_Hidden=100, nr_steps=100):
        self.model = model
        self.top_k_Hidden = top_k_Hidden
        self.nr_steps = nr_steps
        
    def do_intersection_analysis(self):
      vis_lbl_bias, hid_bias=self.model.label_biasing(self.nr_steps) #label biasing
      hidAvg = mean_h_prior(self.model) # hidden biasing

      vettore_indici_allDigits_biasing = torch.empty((0),device= self.model.DEVICE)
      vettore_indici_allDigits_hidAvg = torch.empty((0),device= self.model.DEVICE)

      for cl in range(self.model.Num_classes): #per ogni class
        hid_vec_B = hid_bias[:,cl] #questo è l'hidden state ottenuto con il label biasing di un certo digit
        hid_vec_HA = torch.squeeze(hidAvg[cl]) # hidden state medio a step 1 di un certo digit (hidden biasing)
        top_values_biasing, top_idxs_biasing = torch.topk(hid_vec_B, self.top_k_Hidden) #qui e la linea sotto  trovo i top p indici in termini di attività
        top_values_hidAvg, top_idxs_hidAvg = torch.topk(hid_vec_HA, self.top_k_Hidden)

        vettore_indici_allDigits_biasing = torch.cat((vettore_indici_allDigits_biasing,top_idxs_biasing),0) #concateno i top p indici di ciascun i digits in questo vettore
        vettore_indici_allDigits_hidAvg = torch.cat((vettore_indici_allDigits_hidAvg,top_idxs_hidAvg),0)

      unique_idxs_biasing,count_unique_idxs_biasing = torch.unique(vettore_indici_allDigits_biasing,return_counts=True) #degli indici trovati prendo solo quelli non ripetuti
      unique_idxs_hidAvg,count_unique_idxs_hidAvg = torch.unique(vettore_indici_allDigits_hidAvg,return_counts=True)

      #common_el_idxs_hidAvg = torch.empty((0),device= self.model.DEVICE)
      #common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

      digit_digit_common_elements_count_biasing = torch.zeros((self.model.Num_classes,self.model.Num_classes))
      digit_digit_common_elements_count_hidAvg = torch.zeros((self.model.Num_classes,self.model.Num_classes))

      self.unique_H_idxs_biasing = unique_idxs_biasing
      self.unique_H_idxs_hidAvg = unique_idxs_hidAvg

      result_dict_biasing ={}
      result_dict_hidAvg ={}


      #itero per ogni digit per calcolare le entrate delle matrici 10 x 10
      for row in range(self.model.Num_classes): 
        for col in range(self.model.Num_classes):

          common_el_idxs_hidAvg = torch.empty((0),device= self.model.DEVICE)
          common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

          counter_biasing = 0
          for id in unique_idxs_biasing: #per ogni indice unico del biasing di ogni digit
            digits_found = torch.floor(torch.nonzero(vettore_indici_allDigits_biasing==id)/self.top_k_Hidden)
            #nella linea precedente torch.nonzero(vettore_indici_allDigits_biasing==id) trova le posizioni nell'array vettore_indici_allDigits_biasing
            #che ospitano l'unità ID. ora, essendo che vettore_indici_allDigits_biasing contiene le prime 100 unità più attive di ciascun digit, se divido gli indici per 100
            #trovo per quali digit l'unità ID era attiva
            if torch.any(digits_found==row) and torch.any(digits_found==col): #se i digits trovati ospitano sia il digit riga che quello colonna...
                common_el_idxs_biasing = torch.hstack((common_el_idxs_biasing,id)) #aggiungi ID al vettore di ID che verranno usati per fare biasing
                counter_biasing += 1

          result_dict_biasing[str(row)+','+str(col)] = common_el_idxs_biasing
          digit_digit_common_elements_count_biasing[row,col] = counter_biasing

          counter_hidAvg = 0
          for id in unique_idxs_hidAvg:
            digits_found = torch.floor(torch.nonzero(vettore_indici_allDigits_hidAvg==id)/self.top_k_Hidden)
            if torch.any(digits_found==row) and torch.any(digits_found==col):
                common_el_idxs_hidAvg = torch.hstack((common_el_idxs_hidAvg,id))
                counter_hidAvg += 1
          result_dict_hidAvg[str(row)+','+str(col)] = common_el_idxs_hidAvg
          digit_digit_common_elements_count_hidAvg[row,col] = counter_hidAvg

      self.result_dict_biasing = result_dict_biasing
      self.result_dict_hidAvg = result_dict_hidAvg


      print(digit_digit_common_elements_count_biasing)
      print(digit_digit_common_elements_count_hidAvg)
      #lbl_bias_freqV = digit_digit_common_elements_count_biasing.view(100)/torch.sum(digit_digit_common_elements_count_biasing.view(100))
      #avgH_bias_freqV = digit_digit_common_elements_count_hidAvg.view(100)/torch.sum(digit_digit_common_elements_count_hidAvg.view(100))

      #print(scipy.stats.chisquare(lbl_bias_freqV, f_exp=avgH_bias_freqV))


      return digit_digit_common_elements_count_biasing, digit_digit_common_elements_count_hidAvg



    def generate_chimera_lbl_biasing(self,VGG_cl, elements_of_interest = [8,2],temperature=1, nr_of_examples = 1000, plot=0, entropy_correction=1):
      b_vec =torch.zeros(nr_of_examples,self.model.layersize[0])
      if not(elements_of_interest =='rand'):
        dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1])
        b_vec[:,self.result_dict_biasing[dictionary_key].long()]=1

      else: #write 'rand' in elements of interest
        for i in range(nr_of_examples):
          n1 = random.randint(0, self.model.Num_classes-1)
          n2 = random.randint(0, self.model.Num_classes-1)
          dictionary_key = str(n1)+','+str(n2)
          b_vec[i,self.result_dict_biasing[dictionary_key].long()]=1

      b_vec = torch.unsqueeze(b_vec,2)
      #b_vec = torch.unsqueeze(b_vec,0)
      
      d= self.model.reconstruct_from_hidden(b_vec, self.nr_steps,temperature=temperature)
      
      d = Classifier_accuracy(d, VGG_cl, self.model, plot=plot, Thresholding_entropy=entropy_correction)
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,self.model, Plot=plot, Ian=1)
      
      if nr_of_examples < 16:
          Reconstruct_plot(b_vec, self.model, nr_steps=self.nr_steps, d_type='hidden',temperature=temperature)
      
      return d, df_average,df_sem, Transition_matrix_rowNorm


def Chimeras_nr_visited_states(model, VGG_cl, Ian =[], topk=149, apprx=1,plot=1,compute_new=1, nr_sample_generated =100, entropy_correction=[], cl_labels=[], lS=20):
    def save_mat_xlsx(my_array, filename='my_res.xlsx'):
        # create a pandas dataframe from the numpy array
        my_dataframe = pd.DataFrame(my_array)

        # save the dataframe as an excel file
        my_dataframe.to_excel(filename, index=False)
        # download the file
        files.download(filename)

    n_digits = model.Num_classes
    if Ian!=[]:
      fN='Visited_digits_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr='Visited_digits_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fN_NDST='Nondigit_stateTime_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
    else:
      fN='Visited_digits_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr='Visited_digits_Lbiasing_error_k' + str(topk)+'.xlsx'
      fN_NDST='Nondigit_stateTime_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_Lbiasing_error_k' + str(topk)+'.xlsx'

    if compute_new==1:
      #both
      Vis_states_mat = np.zeros((n_digits, n_digits))
      Vis_states_err = np.zeros((n_digits, n_digits))
      if n_digits==10:
        Non_digit_mat  = np.zeros((n_digits, n_digits))
        Non_digit_err  = np.zeros((n_digits, n_digits))

      if Ian!=[]:
        for row in range(n_digits):
          for col in range(row,n_digits):
            d, df_average,df_sem, Transition_matrix_rowNorm = Ian.generate_chimera_lbl_biasing(VGG_cl,elements_of_interest = [row,col], nr_of_examples = nr_sample_generated, temperature = 1, plot=0, entropy_correction= entropy_correction)
            Vis_states_mat[row,col]=df_average.Nr_visited_states[0]
            Vis_states_err[row,col]=df_sem.Nr_visited_states[0]
            if n_digits==10:
              Non_digit_mat[row,col] = df_average['Non-digit'][0]
              Non_digit_err[row,col] = df_sem['Non-digit'][0]
      else:
        numbers = list(range(n_digits))
        combinations_of_two = list(combinations(numbers, 2))

        for idx, combination in enumerate(combinations_of_two):
          gen_hidden = label_biasing(model, on_digits=  list(combination), topk = topk)
          gen_hidden_rep = gen_hidden.repeat(1,nr_sample_generated)
          d = generate_from_hidden(model, gen_hidden_rep , nr_gen_steps=100, temperature=1, consider_top_k_units = gen_hidden_rep.size()[1], include_energy = 0)
          #d = Classifier_accuracy(d, VGG_cl,model, Thresholding_entropy=entropy_correction, labels=[], Batch_sz= 100, plot=0, dS=30, l_sz=3)
          d = Classifier_accuracy(d, VGG_cl,model, labels=[], Batch_sz= 100, plot=0, dS=30, l_sz=3)
          df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,model,Plot=0,dS=50,Ian=1)
          Vis_states_mat[combination[0],combination[1]]=df_average.Nr_visited_states[0]
          Vis_states_err[combination[0],combination[1]]=df_sem.Nr_visited_states[0]
          if n_digits==10:
            Non_digit_mat[combination[0],combination[1]] = df_average['Non-digit'][0]
            Non_digit_err[combination[0],combination[1]] = df_sem['Non-digit'][0]


      save_mat_xlsx(Vis_states_mat, filename=fN)
      save_mat_xlsx(Vis_states_err, filename=fNerr)
      if n_digits==10:
        save_mat_xlsx(Non_digit_mat, filename=fN_NDST)
        save_mat_xlsx(Non_digit_err, filename=fNerr_NDST)

    else: #load already computed Vis_states_mat
      Vis_states_mat = pd.read_excel(fN)
      Vis_states_err = pd.read_excel(fNerr)
      # Convert the DataFrame to a NumPy array
      Vis_states_mat = Vis_states_mat.values
      Vis_states_err = Vis_states_err.values

      if n_digits==10:
        Non_digit_mat = pd.read_excel(fN_NDST)
        Non_digit_err = pd.read_excel(fNerr_NDST)
        # Convert the DataFrame to a NumPy array
        Non_digit_mat = Non_digit_mat.values
        Non_digit_err = Non_digit_err.values

    if plot==1:

      Vis_states_mat = Vis_states_mat.round(apprx)
      Vis_states_err = Vis_states_err.round(apprx)

      plt.figure(figsize=(15, 15))
      mask = np.triu(np.ones_like(Vis_states_mat),k=+1) # k=+1 per rimuovere la diagonale
      # Set the lower triangle to NaN
      Vis_states_mat = np.where(mask==0, np.nan, Vis_states_mat)
      Vis_states_mat = Vis_states_mat.T
      #ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=False,square=True, cbar=False)
      ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=True, annot_kws={"size": lS},square=True,cbar_kws={"shrink": .82}, fmt='.1f', cmap='jet')
      if not(cl_labels==[]):
        ax.set_xticklabels(cl_labels)
        ax.set_yticklabels(cl_labels)

      #ax.set_xticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=lS)

      plt.xlabel('Class', fontsize = 25) # x-axis label with fontsize 15
      plt.ylabel('Class', fontsize = 25) # y-axis label with fontsize 15
      #cbar = plt.gcf().colorbar(ax.collections[0], location='left', shrink=0.82)
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=lS)
      plt.show()
    if n_digits==10:
      return Vis_states_mat, Vis_states_err,Non_digit_mat,Non_digit_err
    else:
      return Vis_states_mat, Vis_states_err