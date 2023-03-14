import torch
import random
import VGG_MNIST
import plotting_functions
from VGG_MNIST import *
from plotting_functions import *
from google.colab import files
import pandas as pd

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

      for digit in range(self.model.Num_classes): #per ogni digit
        hid_vec_B = hid_bias[:,digit] #questo è l'hidden state ottenuto con il label biasing di un certo digit
        hid_vec_HA = torch.squeeze(hidAvg[digit]) # hidden state medio a step 1 di un certo digit (hidden biasing)
        top_values_biasing, top_idxs_biasing = torch.topk(hid_vec_B, self.top_k_Hidden) #qui e la linea sotto  trovo i top p indici in termini di attività
        top_values_hidAvg, top_idxs_hidAvg = torch.topk(hid_vec_HA, self.top_k_Hidden)

        vettore_indici_allDigits_biasing = torch.cat((vettore_indici_allDigits_biasing,top_idxs_biasing),0) #concateno i top p indici di ciascun i digits in questo vettore
        vettore_indici_allDigits_hidAvg = torch.cat((vettore_indici_allDigits_hidAvg,top_idxs_hidAvg),0)

      unique_idxs_biasing,count_unique_idxs_biasing = torch.unique(vettore_indici_allDigits_biasing,return_counts=True) #degli indici trovati prendo solo quelli non ripetuti
      unique_idxs_hidAvg,count_unique_idxs_hidAvg = torch.unique(vettore_indici_allDigits_hidAvg,return_counts=True)

      #common_el_idxs_hidAvg = torch.empty((0),device= self.model.DEVICE)
      #common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

      digit_digit_common_elements_count_biasing = torch.zeros((10,10))
      digit_digit_common_elements_count_hidAvg = torch.zeros((10,10))

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
      lbl_bias_freqV = digit_digit_common_elements_count_biasing.view(100)/torch.sum(digit_digit_common_elements_count_biasing.view(100))
      avgH_bias_freqV = digit_digit_common_elements_count_hidAvg.view(100)/torch.sum(digit_digit_common_elements_count_hidAvg.view(100))

      print(scipy.stats.chisquare(lbl_bias_freqV, f_exp=avgH_bias_freqV))


      return digit_digit_common_elements_count_biasing, digit_digit_common_elements_count_hidAvg



    def generate_chimera_lbl_biasing(self,VGG_cl, elements_of_interest = [8,2],temperature=1, nr_of_examples = 1000, plot=0):
      b_vec =torch.zeros(nr_of_examples,1000)
      if not(elements_of_interest =='rand'):
        dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1])
        b_vec[:,self.result_dict_biasing[dictionary_key].long()]=1

      else: #write 'rand' in elements of interest
        for i in range(nr_of_examples):
          n1 = random.randint(0, 9)
          n2 = random.randint(0, 9)
          dictionary_key = str(n1)+','+str(n2)
          b_vec[i,self.result_dict_biasing[dictionary_key].long()]=1

      b_vec = torch.unsqueeze(b_vec,2)
      #b_vec = torch.unsqueeze(b_vec,0)
      
      d= self.model.reconstruct_from_hidden(b_vec, self.nr_steps,temperature=temperature)

      
      reconstructed_imgs=d['vis_states']
      d_cl = Classifier_accuracy(reconstructed_imgs, VGG_cl, self.model, plot=plot)
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d_cl,self.model, Plot=plot)
      
      if nr_of_examples < 16:
          Reconstruct_plot(b_vec, self.model, nr_steps=self.nr_steps, d_type='hidden',temperature=temperature)
      
      return d, df_average,df_sem, Transition_matrix_rowNorm

def Chimeras_nr_visited_states(Ian,VGG_cl,apprx=1,plot=1,compute_new=1):
    n_digits = Ian.model.Num_classes
    fN='Visited_digits_k' + str(Ian.top_k_Hidden)+'.xlsx'
    fNerr='Visited_digits_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
    def save_mat_xlsx(my_array, filename='my_res.xlsx'):
        # create a pandas dataframe from the numpy array
        my_dataframe = pd.DataFrame(my_array)

        # save the dataframe as an excel file
        my_dataframe.to_excel(filename, index=False)
        # download the file
        files.download(filename)

    if compute_new==1:
      Vis_states_mat = np.zeros((n_digits, n_digits))
      Vis_states_err = np.zeros((n_digits, n_digits))

      for row in range(n_digits):
        for col in range(row,n_digits):
          d, df_average,df_sem, Transition_matrix_rowNorm = Ian.generate_chimera_lbl_biasing(VGG_cl,elements_of_interest = [row,col], nr_of_examples = 1000, temperature = 1, plot=0)
          Vis_states_mat[row,col]=df_average.Nr_visited_states[0]
          Vis_states_err[row,col]=df_sem.Nr_visited_states[0]

      save_mat_xlsx(Vis_states_mat, filename=fN)
      save_mat_xlsx(Vis_states_err, filename=fNerr)

    else: #load already computed Vis_states_mat
      Vis_states_mat = pd.read_excel(fN)
      # Convert the DataFrame to a NumPy array
      Vis_states_mat = Vis_states_mat.values

      Vis_states_err = pd.read_excel(fNerr)
      # Convert the DataFrame to a NumPy array
      Vis_states_err = Vis_states_err.values

    if plot==1:

      Vis_states_mat = Vis_states_mat.round(apprx)
      Vis_states_err = Vis_states_err.round(apprx)

      plt.figure(figsize=(15, 15))
      mask = np.triu(np.ones_like(Vis_states_mat))
      # Set the lower triangle to NaN
      Vis_states_mat = np.where(mask==0, np.nan, Vis_states_mat)
      ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=False,square=True, cbar_kws={"shrink": .82})
      #ax.set_xticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=20)

      for i in range(n_digits):
          for j in range(n_digits):
              value = Vis_states_mat[i, j]
              error = Vis_states_err[i, j]
              ax.annotate(f'{value:.2f} \n ±{error:.2f}', xy=(j+0.5, i+0.5), 
                          ha='center', va='center', color='white', fontsize=20)

      plt.xlabel('To', fontsize = 25) # x-axis label with fontsize 15
      plt.ylabel('From', fontsize = 25) # y-axis label with fontsize 15
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=20)
      plt.show()

    return Vis_states_mat, Vis_states_err