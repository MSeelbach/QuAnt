
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:15:06 2021

@author: aseelbac
"""
import torch
import numpy as np
import pandas as pd
import matplotlib


import matplotlib.pyplot as plt
from typing import Tuple
import math
import random

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn.functional as F



import Tilings

from model.MLPmodel import Network
from data.Dataset import Dataset


import dimod
import dwave_networkx as dnx
from dwave.system import FixedEmbeddingComposite
import networkx as nx

import pickle 

import neal

width = 78 # width of the neural network
UseDeeperNetwork=True # Use the neural network with L=3 or L=5?
TrainOnWillow=False  # Train on the Willow dataset or on the RandGraph dataset?
UseQuantumAnnealer=False

dim= 4 
Cutoff=int( np.ceil(np.log2 (dim))*dim)



saveStr=''

ToExecute= Tilings.TilingCom()
embeddings, tile = ToExecute.GetTilingEmbedding()

capacity= len(tile.nodes)

mask= torch.zeros( (capacity, capacity) )



for e in tile.edges:
    
     if e[0]<Cutoff and e[0]<Cutoff:
         mask[e[0], e[1]]=1 
         mask[e[1], e[0]]=1 

for n in tile.nodes:
    mask[n,n]=1 

embeddings= embeddings[:141]
BATCH_SIZE = len(embeddings)

EPOCHS=150


def BinaryToPermutation(binEnc,dim):
    # Get a permutation table from the binary encoding
    
    permutationTable= []
    currentNr=[]
    for k in range(len(binEnc)):
        currentNr.append(binEnc[k])
        if (k+1)%(int(np.ceil( np.log2(dim)))) ==0:
            
           
            
            permutationTable.append( int("".join(str(i) for i in currentNr),2))
            
            currentNr=[]

    
    return permutationTable

    
#Energy with collumn major or row major vectorization
def GetEnergy(permutation, Matrix  ):
    
    n= int( np.sqrt(Matrix .shape[0] ))
    
    energy=0
    for k in range(n) :
        for l in range(n):
            energy+=  Matrix[permutation[k]*n + k,  permutation[l]*n + l ]
    
    
    
    return energy


def GetEnergy2(permutation, Matrix  ):
    
    n= int( np.sqrt(Matrix .shape[0] ))
    
    energy=0
    for k in range(n) :
        for l in range(n):
            energy+=  Matrix[k*n + permutation[k],  l*n + permutation[l] ]
    
    
    return energy





Faktor=40

dataFolder= 'data/RandGraph'
if TrainOnWillow:
    dataFolder= 'data/WillowGraphmatching'

#df = genData( BATCH_SIZE* Faktor, dim) #Datasets were build in this way
df = pd.read_pickle(dataFolder+'/trainingData'+str(dim)+'.p')
data_train = DataLoader(dataset = Dataset(df), batch_size = BATCH_SIZE, shuffle =True)

#df2 = genData( BATCH_SIZE* Faktor, dim)
df2 = pd.read_pickle(dataFolder+'/testData'+str(dim)+'.p')
data_test = DataLoader(dataset = Dataset(df2), batch_size = BATCH_SIZE, shuffle =True)
    

#For diferent assignments to the hardware graph operates on indizes from 0 to 8
def umsort(inp):
    
    return   inp%2 *4 + inp//2 

def sortback(inp):
    
    return inp//4  + inp%4 *2



def isPermutation(dim,toTest):
    #Is the vector in binary encoding a valid permutation?
    a= [False]*dim
    currentNr=[]
    for k in range(len(toTest)):
        currentNr.append(toTest[k])
        if (k+1)%(int(np.ceil( np.log2(dim)))) ==0:
            
            if int("".join(str(i) for i in currentNr),2)>=dim:
                break
            
            a[ int("".join(str(i) for i in currentNr),2)]=True
            
            currentNr=[]
    for t in a:
        if t==False:
            return False
        
    return True
    

def quadrEnergyFun(mat,vector):
    
    #Compute vector.T mat vector in torch. Not for the whole batch
    
    matDim= mat.shape[1]
  
    x_gt = torch.reshape(vector, (1,matDim))
   
    x_gt_trans = torch.transpose(x_gt, 0,1)
   
    result = torch.matmul(mat,x_gt_trans )
    

    result1 = torch.matmul(x_gt,result)
    
    return  result1






def DictoArray( dimension, sample,W ):
    # Samplers return dictionaries. Here also the embedding is reversed.
    isPerm=[]
    energies=[0]*BATCH_SIZE
    output=np.zeros(dimension*BATCH_SIZE)
    for place, embedding in enumerate(embeddings):
        checkPerm=[]
        for k in range(dimension):
            output[place *dimension + k  ]= sample[embedding[k][0]] 
            checkPerm.append(sample[embedding[k][0]])
        if True: #isPermutation(dim, checkPerm): #Use commented line if you actively want to distinguish between permutations and not permutations
            isPerm.append(True)
            torchPerm= torch.tensor(np.array(checkPerm, dtype= np.double)).to(device)
            energies[place]= quadrEnergyFun(  W[place],torchPerm)
        else:
            isPerm.append(False)
        
    
    return output, isPerm, energies


def MixTogether( dimension , sample1 , secondBestsample1 , sample2 , energies1,secondBestenergies ,energies2  ):
    
    #Determine sample with lowest and second lowest energy for each of the different problem instances all executed in one run on the sampler
    #energies1: the lowest energies from all the runs that are already mixed together
    #secondBestenergies: the  second-lowest energies from all the runs that are already mixed together
    #energies2: new outputs. Here we need to check if they are better than energies1 or secondBestenergies. If so they will take their place 
    # The samples correspond to the energy-input argument and are ordered in the same way 
    
    energiesOutput=energies1.copy() #New best energies
    energies2Output= secondBestenergies.copy() #New second best energies
    
    output2=secondBestsample1.copy()
    output=sample1.copy()

    for t in range(len(energies1)):
        
            if energies1[t] > energies2[t]:
                   energies2Output[t]= energies1[t]
                   energiesOutput[t]= energies2[t]
                   output2[dimension*t : dimension*(t+1)]= output[dimension*t : dimension*(t+1)]
                   output[dimension*t : dimension*(t+1)]= sample2[dimension*t : dimension*(t+1)]
    
            elif energies1[t] <energies2[t]:
                if energies1[t]== secondBestenergies[t]: # It might happen that a second solution was not found yet.
                    energies2Output[t]  =energies2[t]                  
                    output2[dimension*t : dimension*(t+1)]= sample2[dimension*t : dimension*(t+1)]
                else:
                    if energies2[t]<secondBestenergies[t]:
                        energies2Output[t]  =energies2[t]                  
                        output2[dimension*t : dimension*(t+1)]= sample2[dimension*t : dimension*(t+1)]
    
    
    return output, output2 ,  energiesOutput, energies2Output 




def getGroundStateSim( W, descr):

    nl= int(W.shape[1])
    W.reshape([-1,nl,nl])
    
    number1 = W.shape[0]

    if UseQuantumAnnealer:
        sampleSet= ToExecute.runrealDeal( F.pad(input=W, pad=(0, capacity- int( dim* np.ceil( np.log2(dim))), 0,capacity- int( dim* np.ceil( np.log2(dim))),0,0 ), mode='constant', value=0)
                                          ,descr)    
    else:
        sampleSet= ToExecute.runSimulated( F.pad(input=W, pad=(0, capacity- int( dim* np.ceil( np.log2(dim))), 0,capacity- int( dim* np.ceil( np.log2(dim))),0,0 ), mode='constant', value=0)
                                          )#,descr)

  
    first=True
    bestenergies=[]

    bestenergies2=[]

    
    firstSampleCand=[]
    secondSampleCand=[]
    
    for datum in sampleSet.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
                   if first:
                       firstSampleCand ,isPerm, bestenergies=  DictoArray((dim* int(np.ceil( np.log2(dim)))) , datum.sample,W)
                       secondSampleCand= firstSampleCand
                       bestenergies2=bestenergies
                       first=False
                   else:
                       
                       SampleCand2 ,isPerm2, energies2 = DictoArray((dim* int(np.ceil( np.log2(dim)))) , datum.sample,W)
                       firstSampleCand, secondSampleCand , bestenergies,bestenergies2= MixTogether(dim* int(np.ceil( np.log2(dim))),firstSampleCand,secondSampleCand,SampleCand2,bestenergies,bestenergies2,energies2 ) 
                      
                   
  
    output = firstSampleCand.reshape((number1, dim* int(np.ceil( np.log2(dim)))))

    output2= secondSampleCand.reshape((number1, dim* int(np.ceil( np.log2(dim)))))

    return output, output2





def quadrEnergy(mat,vector):
    #Compute vector.T mat vector in torch. For the whole batch
    
    matDim= mat.shape[2]
  
    x_gt = torch.reshape(vector, (-1,matDim,1))
    
    x_gt_trans = torch.transpose(x_gt, 1,2)
    result = torch.matmul(x_gt_trans, mat)
    result1 = torch.matmul(result,x_gt)
    
    return  result1






def mat_loss_func3(A_mat, x_gt, lamb1, lamb2, network, input, descr):

   #loss function as described in the paper

  A_mat = A_mat.to(device)
  
  
  res_value, sec_value = getGroundStateSim(A_mat,descr)
  
  
  x_gt.type(torch.FloatTensor)
  reshape_size = min(x_gt.shape[0], BATCH_SIZE)
  

 
  res_value = torch.from_numpy(res_value) 

  xsim = torch.reshape(res_value.cuda(), (reshape_size,dim* int(np.ceil( np.log2(dim))),1)).double()
  xsim_trans = torch.transpose(xsim, 1,2)

 

  energy = torch.matmul(xsim_trans, A_mat)
  
  energy = torch.matmul(energy, xsim)


  sec_value = torch.from_numpy(sec_value) 

  xsec = torch.reshape(sec_value.cuda(), (reshape_size,dim* int(np.ceil( np.log2(dim))),1)).double()
  xsec_trans = torch.transpose(xsec, 1,2)

 

  energy2 = torch.matmul(xsec_trans, A_mat)
  
  energy2 = torch.matmul(energy2, xsec)
 
  
  
  gtenergy=quadrEnergy(A_mat, (x_gt).double() )
  
  result =  gtenergy-energy  -lamb1*torch.abs(gtenergy-energy2)+ lamb2 * network.sparse_loss(input)   #-energyLowest.cuda() + lamb1 * network.sparse_loss(input) #- difference * lamb2 * torch.abs(result1 - res_value[:,1])

  difference = torch.clone(torch.flatten(result, start_dim=1))
  difference[difference != 0] = -1
  difference[difference == 0] = 1
  difference[difference == -1] = 0
  
  

  result = result.mean()
  return result, difference.detach().sum().item()





#Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
mask = mask.to(device)


net = Network(dim, width, False, mask)
net.to(device)


criterion = mat_loss_func3
#criterion = nn.L1Loss() for pure

optm = Adam(net.parameters(), lr=0.00001)
   

difference_stat = []
diff_temp = 0
#net.load_state_dict(torch.load('images/width78Save24'))

for epoch in range(0,150):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(data_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optm.zero_grad()
        
        
        inputs.double()
        net.double()
        # forward + backward + optimize
        outputs = net(inputs)
        loss, diff_sum = criterion(outputs, labels,  10e-3, 10e-4, net, inputs,'epoch'+str(epoch)+'I'+str(i))
        diff_temp += diff_sum

        loss.backward()
        optm.step()



        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    
            diff_temp = diff_temp / 20
            difference_stat.append(diff_temp)
            diff_temp = 0
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            
            saveStr+= str('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 20))
            saveStr+= '\n'
            running_loss = 0.0


#At the end evaluate on the test data
correct = 0
total = 0
histogram = np.zeros( int( dim* np.ceil( np.log2(dim)))+1)
zero = 0
A_distripution = np.array([])
with torch.no_grad():
    counter = 0
    for data in data_test:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        predicted = outputs.data
        conversion = predicted.cpu()
        
        labels_numpy = labels.cpu().numpy()
        total += labels.size(0) 
       
        res_angle, secondStuff = getGroundStateSim( outputs, 'test'+'epoch'+str(EPOCHS) )
    
        counter +=1
        hist = np.sum(np.abs(res_angle - labels_numpy), axis=1 ) 
        hist = hist.astype(int)
        for h in hist:
          histogram[h] += 1


print('Accuracy of the network on the test dataset: %d %%' % (
    100 * histogram[0] / total))


saveStr+= str('Accuracy of the network on the test dataset: %d %%' % (
    100 * histogram[0] / total))
saveStr+= '\n'
y_pos = np.arange( int( dim* np.ceil( np.log2(dim)))+1)
fig, ax = plt.subplots()

ax.bar(y_pos, histogram)

plt.xlabel('Hamming distance to Ground Truth')

plt.ylabel('Number of occurence')

plt.title(str(epoch)+ ' Epochs')

plt.savefig('images/'+'width'+str(width)+str(EPOCHS)+'EpochsSigmoidMinusOneHalfInteraction.png')
plt.close('all')
torch.save(net.state_dict(),"images/"+'width'+str(width) +"Save" +str(EPOCHS))
fig, ax = plt.subplots()
im= ax.imshow(conversion[0])
fig.colorbar(im)
plt.savefig('images/'+'width'+str(width)+'picture'+str(EPOCHS)+'.png')
plt.close('all')




