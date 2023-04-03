# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:06:53 2023

@author: Marcel
"""
import numpy as np

import pandas as pd

import torch


def PermutationToBinary(Perm):
    m= int( np.ceil(np.log2(Perm.shape[0])))
    result=     (((Perm[:,None] & (1 << np.flip(np.arange(m))))) > 0).astype(int)
    result=result.reshape((-1))
    return result


def genData( n , size ):
    
    dic = {'Input' : [], 'Output' : []}

    

    for k in range(n):
        Distances= np.random.uniform(0,1,(size, size)).astype('d')
        
        permutation= np.random.permutation(np.arange(size)) #ground truth permutation
        
        DistancesMixed= Distances.copy()
        
        DistancesMixed[:,:]= DistancesMixed[permutation,:]
        DistancesMixed[:,:]= DistancesMixed[:,permutation]
    
        
        #Difference between distances
        Matrix = np.zeros((size**2, size**2))
        Matrix =np.abs( np.kron( Distances, np.ones((size, size)) )- np.kron(  np.ones((size, size)) ,DistancesMixed)) 
        
   
        
        
        dic['Input'].append(np.ravel(Matrix)) #vectorized matrix with distance differences
        dic['Output'].append(PermutationToBinary(permutation)) #permutation in binary representation
    df = pd.DataFrame(dic)
    df = df.sample(frac=1).reset_index(drop=True)
    return df