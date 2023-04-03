# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:59:57 2023

@author: Marcel
"""

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import math


class Network(nn.Module):

    def __init__(self,dim, width, largerNetwork, mask ):
        super().__init__()
        #dim: dimension of the permutations
        #largerNetwork: boolean if more layers should be used 
        #width: width of the layers
        
   
        self.largerNetwork= largerNetwork    
        self.fc2 = nn.Linear( dim**4  , width)
        if largerNetwork:
            self.fc4 = nn.Linear(width, width)
            self.fc5 = nn.Linear(width +dim**4   , width)
        self.fc6 = nn.Linear(width, width)

        self.fc12 = nn.Linear(width , int( dim* np.ceil( np.log2(dim))))
        self.dim= dim
        self.width= width

        self.mask= mask



    def forward(self,x):
        input_x = x.clone()
        x = F.relu(self.fc2(x))
        
        if self.largerNetwork:
            x = F.relu(self.fc4(x))
            x = torch.cat((x, input_x), dim=1)
            x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))



        x = self.fc12(x)

        
        dim= self.dim
        x = torch.reshape(x, (x.shape[0], int( dim* np.ceil( np.log2(dim)))))
        
        
        return x

    def sparse_loss(self,x):
        loss = 0 
        input_x = x.clone()
        x = F.relu(self.fc2(x))
        loss += torch.mean(torch.abs(x))
        if self.largerNetwork:

            x = F.relu(self.fc4(x))
            loss += torch.mean(torch.abs(x))

            x = torch.cat((x, input_x), dim=1)
            x = F.relu(self.fc5(x))
            loss += torch.mean(torch.abs(x))


        x = F.relu(self.fc6(x))
        loss += torch.mean(torch.abs(x))

      

        x = self.fc12(x)



        dim= self.dim
        x = torch.reshape(x, (x.shape[0], int( dim* np.ceil( np.log2(dim)))))
        
             
       

        return loss