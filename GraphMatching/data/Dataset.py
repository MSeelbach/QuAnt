# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:14:47 2023

@author: Marcel
"""
import torch


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataFrame):
        'Initialization'
        self.df = dataFrame

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.df.index)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.df['Input'][index]
        y = self.df['Output'][index]
        return X, y