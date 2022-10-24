import torch
import VGG_MNIST
from VGG_MNIST import *

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

    def generate_chimera_lbl_biasing(self, VGG_cl, elements_of_interest = [8,2],temperature=1, nr_of_examples = 1000):
      dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1])


      b_vec =torch.zeros(nr_of_examples,1000)
      b_vec[:,self.result_dict_biasing[dictionary_key].long()]=1
      b_vec = torch.unsqueeze(b_vec,2)
      #b_vec = torch.unsqueeze(b_vec,0)
      
      d= self.model.reconstruct_from_hidden(b_vec, self.nr_steps,temperature=temperature)

      
      reconstructed_imgs=d['vis_states']
      d_cl = Classifier_accuracy(reconstructed_imgs, VGG_cl, self.model, plot=0)
      df_average,df_sem = classification_metrics(d_cl,self.model)
      
      if nr_of_examples < 16:
          Reconstruct_plot(b_vec, self.model, nr_steps=self.nr_steps, d_type='hidden',temperature=temperature)
      
      return (d, df_average)

