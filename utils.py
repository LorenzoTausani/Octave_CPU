#Loading MNIST dataset
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import torch
import pickle
import os
import numpy as np
import math
from RBM import RBM
from methods import compute_inverseW_for_lblBiasing
from google.colab import files

def load_MNIST_data(DEVICE):
    
    mnist_data_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose(
                        [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    mnist_data_test = datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.Compose(
                        [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))


    mnist_data_train.data = (mnist_data_train.data.type(torch.FloatTensor)/255)
    mnist_data_test.data = (mnist_data_test.data.type(torch.FloatTensor)/255)

    #https://stackoverflow.com/questions/68056122/attributeerror-cant-set-attribute-in-splitting-mnist-dataset


    #Lets us visualize a number from the data set
    idx = 1
    img = mnist_data_test.data[idx]
    print("The number shown is the number: {}".format(mnist_data_test.train_labels[idx]) )
    plt.imshow(img , cmap = 'gray')
    plt.show()

    train_data = mnist_data_train.data.to(DEVICE)
    train_labels = mnist_data_train.train_labels.to(DEVICE)

    test_data = mnist_data_test.data.to(DEVICE)
    test_labels = mnist_data_test.train_labels.to(DEVICE)

    #modificare solo se necessario
    sample_test_data = test_data[:1000,:] 
    sample_test_labels = test_labels[:1000]

    return train_data, train_labels, sample_test_data, sample_test_labels


def model_load_or_create(train_data, train_labels, DEVICE):
  #Funzione per loaddare o creare un modello RBM
  Load_yn=int(input('do you want to load an old model? (1=yes, 0=no)'))

  if Load_yn==1:
    nr_train_epochs_done=int(input('quante epoche di training della RBM?'))
    vis_act_type = input('Che tipo di visibile? (continous,binary)')
    hid_act_type = input('Che tipo di hidden? (binary, gaussian, ReLU, NReLU)')

    V = 'V'+vis_act_type[0]
    H = 'H'+hid_act_type[0]
    if hid_act_type == 'NReLU':
      H=H+hid_act_type[1]
    nr_steps = 100

    filename = 'OctaveCPU_RBM'+ str(nr_train_epochs_done)+'_'+V+H+'_nr_steps'+str(nr_steps)

    model = load_model(filename)
    if not(hasattr(model, 'weights_inv')):
        compute_inverseW_for_lblBiasing(model, train_data, train_labels)

  else:
    num_epochs = int(input('trainare la rete? quante epoche? (0 se non si vuole trainare'))
    vis_act_type = input('Che tipo di visibile? (continous,binary)')
    hid_act_type = input('Che tipo di hidden? (binary, gaussian, ReLU, NReLU)')
    
    model = RBM(maxepochs= num_epochs ,device=DEVICE, Visible_mode = vis_act_type, Hidden_mode=hid_act_type)    
    model.train(train_data,train_labels)
    compute_inverseW_for_lblBiasing(model, train_data, train_labels)

  save_yn = int(input('salvare il modello? (0=no, 1=si)'))

  if save_yn ==1:
    model.save_model()

  return model

def load_model(filename):

    filename = '/content/gdrive/My Drive/' + filename + '/' + filename + '.pkl'

    from google.colab import drive
    drive.mount('/content/gdrive')

    with open(filename, 'rb') as inp:
        model = pickle.load(inp)

    return model  

def generatedData_load_or_create(model, sample_test_data,nr_steps=100,temperature=1,consider_top_H=1000):
  Load_gen_data=int(input('Which generated data you want to load? (0=none, 1=reconstruction, 2=label biasing, 3=avg hidden biasing, 4= all)'))
  save_path = "/content/gdrive/My Drive/Generations used in thesis"
  Filename_standardReconstruction = save_path +'/'+ 'Reconstruction_sampleTestData_100st_1t_allH.pkl'
  Filename_standardLabelBiasing = save_path +'/'+ 'LabelBiasing_1000samples_100st_1t_allH.pkl'
  Filename_standardAvgHiddenBiasingDigits = save_path +'/'+ 'AvgHiddenBiasingDigits_1000samples_100st_1t_allH.pkl'
  Filename_standardAvgHiddenBiasingGrandMean = save_path +'/'+ 'AvgHiddenBiasingGrandMean_1000samples_100st_1t_allH.pkl'

  if Load_gen_data==1 or Load_gen_data==4:
    with open(Filename_standardReconstruction, 'rb') as inp:
      d_reconstruction = pickle.load(inp) 
  else:
    d_reconstruction = model.reconstruct(sample_test_data, nr_steps, temperature=temperature,consider_top=consider_top_H)
    try:
        os.mkdir(save_path)
    except:
        print("Folder already found")
    
    with open(Filename_standardReconstruction, 'wb') as outp:  # Overwrites any existing file.
      pickle.dump(d_reconstruction, outp, pickle.HIGHEST_PROTOCOL)    

  if Load_gen_data==2 or Load_gen_data==4:
    with open(Filename_standardLabelBiasing, 'rb') as inp:
      d_lbl_bias = pickle.load(inp)
  else:
    vis_lbl_bias, gen_hidden_act=model.label_biasing(nr_steps=nr_steps)
    for n in range(100): #100 is for 1000 total label biasing digits
      if n==0:
        VStack_lblBias = vis_lbl_bias
      else:
        VStack_lblBias = torch.vstack((VStack_lblBias,vis_lbl_bias))
        
    #VStack_labels=torch.tensor(range(10), device = 'cuda')
    #VStack_labels=VStack_labels.repeat(100)
    VStack_lblBias = VStack_lblBias.view((1000,28,28))
    d_lbl_bias= model.reconstruct(VStack_lblBias, nr_steps, temperature=temperature,consider_top=consider_top_H)

    try:
        os.mkdir(save_path)
    except:
        print("Folder already found")
    
    with open(Filename_standardLabelBiasing, 'wb') as outp:  # Overwrites any existing file.
      pickle.dump(d_lbl_bias, outp, pickle.HIGHEST_PROTOCOL)

  if Load_gen_data == 3 or Load_gen_data==4:
    with open(Filename_standardAvgHiddenBiasingDigits, 'rb') as inp:
      d_AvgHidBias_digits = pickle.load(inp)
    with open(Filename_standardAvgHiddenBiasingGrandMean, 'rb') as inp:
      d_AvgHidBias_grandMean = pickle.load(inp)
  else:
    Avg_hid_grandMean = mean_h_prior(model)[10:,:,:]
    Avg_hid_digits = mean_h_prior(model)[:10,:,:]

    for n in range(100): #100 is for 1000 total label biasing digits
      if n==0:
        VStack_avgHid_grandMean = Avg_hid_grandMean
        VStack_avgHid_digits = Avg_hid_digits
      else:
        VStack_avgHid_grandMean = torch.vstack((VStack_avgHid_grandMean,Avg_hid_grandMean))
        VStack_avgHid_digits = torch.vstack((VStack_avgHid_digits,Avg_hid_digits))

    d_AvgHidBias_grandMean= model.reconstruct_from_hidden(VStack_avgHid_grandMean , nr_steps, temperature=temperature, consider_top=consider_top_H) #faccio la ricostruzione da hidden
    d_AvgHidBias_digits= model.reconstruct_from_hidden(VStack_avgHid_digits , nr_steps, temperature=temperature, consider_top=consider_top_H) #faccio la ricostruzione da hidden
    try:
        os.mkdir(save_path)
    except:
        print("Folder already found")
    
    with open(Filename_standardAvgHiddenBiasingDigits, 'wb') as outp:  # Overwrites any existing file.
      pickle.dump(d_AvgHidBias_digits, outp, pickle.HIGHEST_PROTOCOL)

    with open(Filename_standardAvgHiddenBiasingGrandMean, 'wb') as outp:  # Overwrites any existing file.
      pickle.dump(d_AvgHidBias_grandMean, outp, pickle.HIGHEST_PROTOCOL)

  return d_reconstruction,d_lbl_bias, d_AvgHidBias_digits,d_AvgHidBias_grandMean

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

def cos_temp_generator(delta=0.5, zero=0.9, frequency=10, Plot=1, n_steps=100):
  x = np.linspace(0, n_steps+frequency, n_steps+frequency+1)
  y = zero+delta*np.cos((math.pi/frequency)*x)
  temp_sc = y[frequency:]
  x = np.linspace(0, n_steps, n_steps+1)

  if Plot==1:
    plt.figure(figsize = (8, 6))
    plt.plot(x, temp_sc, 'b')
    plt.ylabel('Temperature')
    plt.xlabel('Reconstruction step')
    plt.show()
  temp_sc = temp_sc.tolist()
  
  return temp_sc

def error_propagation(measures, measures_error, operation = 'average'):
  #https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
  nr_of_measures = len(measures_error)
  if not(isinstance(measures, np.ndarray)):
    measures = np.asarray(measures)
  if not(isinstance(measures_error, np.ndarray)):
    measures_error = np.asarray(measures_error)

  if operation=='average':
    propagated_err = (1/nr_of_measures)*np.sqrt(np.sum(np.power((measures_error),2))) #senza radice quadrata?
  elif operation == 'ratio':
    propagated_err = np.sum(np.power(measures_error/measures,2))*(measures[0]/measures[1]) #i.e. the result of the ratio
  elif operation =='sum/diff':
    propagated_err = np.sqrt(np.sum(np.power((measures_error),2)))

  

  return propagated_err


def save_mat_xlsx(my_array, filename='my_res.xlsx'):
    # create a pandas dataframe from the numpy array
    my_dataframe = pd.DataFrame(my_array)

    # save the dataframe as an excel file
    my_dataframe.to_excel(filename, index=False)
    # download the file
    files.download(filename)