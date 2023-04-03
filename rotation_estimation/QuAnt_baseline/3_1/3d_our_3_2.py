import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from numpy import linalg as LA
import gc
import os
import glob
import random
import sys

BATCH_SIZE = 32
EPOCHS = 20
TOTAL_RESULT_VECTOR = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
AMOUNT_OF_WRONG_PERMUTATIONS_ls = [0.01, 0.05, 0.1, 0.15, 0.2]

AMOUNT_OF_WRONG_PERMUTATIONS = 0.0 #AMOUNT_OF_WRONG_PERMUTATIONS_ls[int(sys.argv[-1]) - 1]

"""The return matrix is a simple rotational matrix defined like [here](https://en.wikipedia.org/wiki/Rotation_matrix). The angle $\theta \in [0,2\pi]$ is mapped to the set of $\{0, \dots, 2^9 - 1\}$ Here $\pi$ will be mapped to 0. Every angle between $\pi$ and $0$ will have values with a leading 0 and every angle between $\pi$ and $2 \pi$ will have a leading 1

The generated point could will be normally distributed around the origin with $m$ points
"""

def gen_PointCloud(num_of_points):
  res = np.ones((num_of_points, 3))
  res = -1 * res 
  return res + 2 * np.random.rand(num_of_points,3)

def quaternion_to_rotation(Q):
    
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = q0 * q0 + q1 * q1 - (q2 * q2 + q3 * q3)
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q0 * q2 + q1 * q3)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = q0 * q0 + q2 * q2 - (q1 * q1 + q3 * q3)
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = q0 * q0 + q3 * q3 - (q1 * q1 + q2 * q2)
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    
    return rot_matrix

def euler_from_quaternion(Q):
        x = Q[0]
        y = Q[1]
        z = Q[2]
        w = Q[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def calc_rot_mat_from_angle(psi, varphi, phi, tensorTorch = True):
  r11 = np.cos(varphi) * np.cos(psi)
  r12 = np.cos(varphi) * np.sin(psi)
  r13 =  - np.sin(varphi)

  r21 = - np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(varphi) * np.cos(psi)
  r22 = np.cos(phi) * np.cos(psi) + np.sin(psi) * np.sin(varphi) * np.sin(phi)
  r23 = np.sin(phi) * np.cos(varphi)

  r31 =  np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(varphi) * np.cos(psi)
  r32 =  - np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(varphi) * np.sin(psi)
  r33 = np.cos(phi) * np.cos(varphi)

  res =  np.array([[r11, r21, r31],[r12, r22,r32],[r13,r23,r33]])
  if tensorTorch:
    #print(res)
    return torch.tensor(res)
  else:
    return res

def gen_rotated_image(P):
  q3 = np.random.rand(4)
  q3 = (1/LA.norm(q3)) * q3 
  #q3 = (16 * np.pi/ 180) * q3
  a1, a2, a3 = euler_from_quaternion(q3)
  #a1 =   20 * np.pi / 180  * (a1 * (2 / np.pi))
  #a2 =   10 * np.pi / 180  * (a2 * (2 / np.pi))
  #a3 =   20 * np.pi / 180  * (a3 * (2 / np.pi))
  a1 =   20 * np.pi / 180  * (a1 / (2 * np.pi))
  a2 =   10 * np.pi / 180  * (a2 / (2 * np.pi))
  a3 =   20 * np.pi / 180  * (a3 / (2 * np.pi))

  rot_mat = calc_rot_mat_from_angle(a1,a2,a3, tensorTorch=False)
  
  a = range(P.shape[0])
  b = random.sample(a, int(P.shape[0] * AMOUNT_OF_WRONG_PERMUTATIONS))
  c = set(a).difference(set(b))
  c = list(c)
  index = b + c
  P_per = P[index,:]

  result = np.transpose(rot_mat @ np.transpose(P_per))
  return result, a1, a2, a3, rot_mat

def gen_rotated_image_fst_axis(P):
  q3 = np.random.rand(4)
  q3 = (1/LA.norm(q3)) * q3 
  #q3 = (16 * np.pi/ 180) * q3
  a1, a2, a3 = euler_from_quaternion(q3)
  #a1 =   20 * np.pi / 180  * (a1 * (2 / np.pi))
  #a2 =   10 * np.pi / 180  * (a2 * (2 / np.pi))
  #a3 =   20 * np.pi / 180  * (a3 * (2 / np.pi))
  a1 =   20 * np.pi / 180  * (a1 / (2 * np.pi))
  a2 =   10 * np.pi / 180  * (a2 / (2 * np.pi))
  a3 =   20 * np.pi / 180  * (a3 / (2 * np.pi))

  rot_mat = calc_rot_mat_from_angle(a1,a2,a3, tensorTorch=False)

  a = range(P.shape[0])
  b = random.sample(a, int(P.shape[0] * AMOUNT_OF_WRONG_PERMUTATIONS))
  c = set(a).difference(set(b))
  c = list(c)
  index = b + c
  P_per = P[index,:]

  result = np.transpose(rot_mat @ np.transpose(P_per))

  return result, a1, a2, a3, rot_mat

def gen_rotated_image_snd_axis(P):
  q3 = np.random.rand(4)
  q3 = (1/LA.norm(q3)) * q3 
  #q3 = (16 * np.pi/ 180) * q3
  a1, a2, a3 = euler_from_quaternion(q3)
  #a1 =   20 * np.pi / 180  * (a1 * (2 / np.pi))
  #a2 =   10 * np.pi / 180  * (a2 * (2 / np.pi))
  #a3 =   20 * np.pi / 180  * (a3 * (2 / np.pi))
  a1 =  0
  a2 =   10 * np.pi / 180  * (a2 / (2 * np.pi))
  a3 =   20 * np.pi / 180  * (a3 / (2 * np.pi))

  rot_mat = calc_rot_mat_from_angle(a1,a2,a3, tensorTorch=False)
  a = range(P.shape[0])
  b = random.sample(a, int(P.shape[0] * AMOUNT_OF_WRONG_PERMUTATIONS))
  c = set(a).difference(set(b))
  c = list(c)
  index = b + c
  P_per = P[index,:]

  result = np.transpose(rot_mat @ np.transpose(P_per))
  return result, a1, a2, a3, rot_mat

def gen_rotated_image_third_axis(P):
  q3 = np.random.rand(4)
  q3 = (1/LA.norm(q3)) * q3 
  #q3 = (16 * np.pi/ 180) * q3
  a1, a2, a3 = euler_from_quaternion(q3)
  #a1 =   20 * np.pi / 180  * (a1 * (2 / np.pi))
  #a2 =   10 * np.pi / 180  * (a2 * (2 / np.pi))
  #a3 =   20 * np.pi / 180  * (a3 * (2 / np.pi))
  a1 =   0
  a2 =   0
  a3 =   20 * np.pi / 180  * (a3 / (2 * np.pi))

  rot_mat = calc_rot_mat_from_angle(a1,a2,a3, tensorTorch=False)
  a = range(P.shape[0])
  b = random.sample(a, int(P.shape[0] * AMOUNT_OF_WRONG_PERMUTATIONS))
  c = set(a).difference(set(b))
  c = list(c)
  index = b + c
  P_per = P[index,:]

  result = np.transpose(rot_mat @ np.transpose(P_per))
  return result, a1, a2, a3, rot_mat

def bin_array(num, m):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

a = torch.randint(high = 8, size = (4,10))

dec2bin(a,3).shape

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    # faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts#, faces

def generate_Data(number_of_rotations, disc_steps, num_image, num_last_img, img_ls):
  ls_img = []
  for j in range(num_image, num_last_img):

    with open(img_ls[j]) as f:
      P = np.array(read_off(f))
      #P = P.transpose()
      mean = P.mean(axis = 0)
      P -= mean
      scale = 1 / P.std(axis = 0) #1 / max(abs(P.min()), abs(P.max())) #
    
      P = scale * P

    for _ in range(number_of_rotations):
      ret, a11, a22, a33, rot_mat = gen_rotated_image_fst_axis(P)
      a1 = torch.tensor(int(2**disc_steps * (math.degrees(a11) +10)/20))
      # changed here, to male the transition smoother and easier for later calc.
      a2 = torch.tensor(int(2**disc_steps * (math.degrees(a22) + 5)/10))
      a3 = torch.tensor(int(2**disc_steps * (math.degrees(a33) + 10)/20))

      #ret = np.matmul(ret.transpose(), P) / len(P)
      a1 = dec2bin(a1, disc_steps)
      a2 = dec2bin(a2, disc_steps)
      a3 = dec2bin(a3, disc_steps)
      final_angle = np.concatenate((a1,a2,a3))
      final_angle = final_angle.astype('f')
      ret = ret.astype('f')
      #ls_img.append([ret.flatten(), a1, [a11,a22,a33]])
      ls_img.append([P, ret, final_angle])


  return pd.DataFrame(ls_img, columns=['Image_1', 'Image_2', 'Output'])    

def generate_Data_fst(number_of_rotations, disc_steps, num_image, num_last_img, img_ls):
  ls_img = []
  for j in range(num_image, num_last_img):

    with open(img_ls[j]) as f:
      P = np.array(read_off(f))
      #P = P.transpose()
      mean = P.mean(axis = 0)
      P -= mean
      scale = 1 / P.std(axis = 0) #1 / max(abs(P.min()), abs(P.max())) #
    
      P = scale * P

    for _ in range(number_of_rotations):
      ret, a11, a22, a33, rot_mat = gen_rotated_image(P)
      a2 = torch.tensor(0)
      a3 = torch.tensor(0)
      a1 = torch.tensor(int(2**disc_steps * (math.degrees(a11) +10)/20))
      # changed here, to male the transition smoother and easier for later calc.


      ret = np.matmul(ret.transpose(), P) / len(P)
      a1 = dec2bin(a1, disc_steps)
      a2 = dec2bin(a2, disc_steps)
      a3 = dec2bin(a3, disc_steps)
      final_angle = a1.cpu().numpy() #np.concatenate((a1,a2,a3))
      final_angle = final_angle.astype('f')
      ret = ret.astype('f')
      #ls_img.append([ret.flatten(), a1, [a11,a22,a33]])
      ls_img.append([ret.flatten(), final_angle, rot_mat])


  return pd.DataFrame(ls_img, columns=['Input', 'Output', 'RotMat'])

def generate_Data_snd(number_of_rotations, disc_steps, num_image, num_last_img, img_ls):
  ls_img = []
  for j in range(num_image, num_last_img):

    with open(img_ls[j]) as f:
      P = np.array(read_off(f))
      #P = P.transpose()
      mean = P.mean(axis = 0)
      P -= mean
      scale = 1 / P.std(axis = 0) #1 / max(abs(P.min()), abs(P.max())) #
    
      P = scale * P

    for _ in range(number_of_rotations):
      ret, _, a22, _, rot_mat = gen_rotated_image_snd_axis(P)
    
      a1 = torch.tensor(0)
      # changed here, to male the transition smoother and easier for later calc.
      a2 = torch.tensor(int(2**disc_steps * (math.degrees(a22) + 5)/10))
      a3 = torch.tensor(0)
      ret = np.matmul(ret.transpose(), P) / len(P)
      a1 = dec2bin(a1, disc_steps)
      a2 = dec2bin(a2, disc_steps)
      a3 = dec2bin(a3, disc_steps)
      final_angle = a2.cpu().numpy() #np.concatenate((a1,a2,a3))
      final_angle = final_angle.astype('f')
      ret = ret.astype('f')
      #ls_img.append([ret.flatten(), a1, [a11,a22,a33]])
      ls_img.append([ret.flatten(), final_angle, rot_mat])


  return pd.DataFrame(ls_img, columns=['Input', 'Output', 'RotMat'])

def generate_Data_third(number_of_rotations, disc_steps, num_image, num_last_img, img_ls):
  ls_img = []
  for j in range(num_image, num_last_img):

    with open(img_ls[j]) as f:
      P = np.array(read_off(f))
      #P = P.transpose()
      mean = P.mean(axis = 0)
      P -= mean
      scale = 1 / P.std(axis = 0) #1 / max(abs(P.min()), abs(P.max())) #
    
      P = scale * P

    for _ in range(number_of_rotations):
      ret, _, _, a33, rot_mat = gen_rotated_image_third_axis(P)
      #a1 = torch.tensor(int(2**disc_steps * (math.degrees(a11) % 16)/16))
      # changed here, to male the transition smoother and easier for later calc.
      #a2 = torch.tensor(int(2**disc_steps * (math.degrees(a22) + 4)/8))
      a1 = torch.tensor(0)
      a2 = torch.tensor(0)
      a3 = torch.tensor(int(2**disc_steps * (math.degrees(a33) + 10)/20))      
      ret = np.matmul(ret.transpose(), P) / len(P)
      a1 = dec2bin(a1, disc_steps)
      a2 = dec2bin(a2, disc_steps)
      a3 = dec2bin(a3, disc_steps)
      final_angle = a3.cpu().numpy()#np.concatenate((a1,a2,a3))
      final_angle = final_angle.astype('f')
      ret = ret.astype('f')
      #ls_img.append([ret.flatten(), a1, [a11,a22,a33]])
      ls_img.append([ret.flatten(), final_angle, rot_mat])


  return pd.DataFrame(ls_img, columns=['Input', 'Output', 'RotMat'])

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

class Dataset_train(torch.utils.data.Dataset):
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
        X = self.df['Image_1'][index]
        Y = self.df['Image_2'][index]
        Z = self.df['Output'][index]
        return X, Y, Z

os.chdir(r'') # dictionary, where ModelNet10 is saved
#os.chdir(r'/ModelNet10')

myFiles = glob.glob('**/**/*.off')
dic = {'Input' : [], 'Output' : []}
random.shuffle(myFiles)

#df = generate_Data_fst(1000, 5, 1,100, myFiles)
#df = generate_Data_fst(5000, 5, 1,300, myFiles)
#df = generate_Data_fst(5, 5, 1,30, myFiles)

#data_train_fst = DataLoader(dataset = Dataset(df), batch_size = BATCH_SIZE, shuffle =False)
#df2 = generate_Data_fst(100, 5, 101, 110, myFiles)

df = generate_Data_snd(1000, 5, 1,500, myFiles)
#df = generate_Data_snd(50, 5, 1,30, myFiles)

data_train_snd = DataLoader(dataset = Dataset(df), batch_size = BATCH_SIZE, shuffle =False)
#df = generate_Data_third(5000, 5, 1,300, myFiles)
#df = generate_Data_third(50, 5, 1,30, myFiles)

#data_train_third = DataLoader(dataset = Dataset(df), batch_size = BATCH_SIZE, shuffle =False)

df_data_test = generate_Data(100, 5, 301, 331, myFiles)
#data_test = DataLoader(dataset = Dataset_train(df2), batch_size = BATCH_SIZE, shuffle =False)
os.chdir(r'') # current working dict

"""In order to generate the dataset we will randomly generate the point cloud and rotate it with a random angle n times. This product $H^T P$ will then be saved as a vector as well as the angle in bits"""

def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()

def gen_all_binary_vectors_fst_dim(length: int) -> torch.Tensor:
    temp = ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()
    return torch.cat((temp, torch.zeros(temp.shape[0], 10)), dim = 1)

def gen_all_binary_vectors_second_dim(length: int) -> torch.Tensor:
    temp = ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()
    return torch.cat((torch.zeros(temp.shape[0], TOTAL_RESULT_VECTOR), temp, torch.zeros(temp.shape[0], TOTAL_RESULT_VECTOR)), dim = 1)

def gen_all_binary_vectors_third_dim(length: int) -> torch.Tensor:
    temp = ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()
    return torch.cat((torch.zeros(temp.shape[0], 10), temp), dim = 1)

def binary(x, bits):
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    result =  x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    result = result.float()
    return result

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc2 = nn.Linear(9, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc12 = nn.Linear(32, TOTAL_RESULT_VECTOR*TOTAL_RESULT_VECTOR)

        #self.fc8 = nn.Linear(128, 256)
        #self.fc9 = nn.Linear(256, 256)
        #self.fc10 = nn.Linear(256, 512)


    def forward(self,x):
        input_x = x.clone()
        input_x = torch.repeat_interleave(input_x, 7, dim = 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))

      
        x = self.fc12(x)

        reshape_size = min(x.shape[0], BATCH_SIZE)

        x = torch.reshape(x, (reshape_size, TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR))
        x_trans = torch.transpose(x,1,2)

        x = torch.sin(x + x_trans)
        return x

    def sparse_loss(self,x):
        loss = 0 
        x = F.relu(self.fc2(x))
        loss += torch.mean(torch.abs(x))
        x = F.relu(self.fc4(x))
        loss += torch.mean(torch.abs(x))

        x = self.fc12(x)
        loss += torch.mean(torch.abs(x))

        return loss

device

def bruteForceAngle(mat, x_gt): 
  
    possible_x_trans = gen_all_binary_vectors(TOTAL_RESULT_VECTOR)
    reshape_size = min(x_gt.shape[0], BATCH_SIZE)
    possible_x_trans = possible_x_trans.repeat(reshape_size,1,1)
    possible_x_trans = possible_x_trans.to(device)
    possible_x = torch.transpose(possible_x_trans,1,2)
    results = torch.matmul(possible_x_trans, mat)
    results = torch.matmul(results, possible_x)
    results = torch.diagonal(results, dim1 = 1, dim2 = 2)
    results = torch.topk(results,2,largest=False,dim=1)
    result_value = results.values
    result_angle = results.indices
    result_angle = possible_x_trans[result_angle]

    a1, a2, a3 = torch.split(result_angle, [TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR], dim=1)

    return result_value, a1, a2, a3   

def bruteForceAngle_fst_dim(mat, x_gt): 
  
    possible_x_trans = gen_all_binary_vectors_fst_dim(TOTAL_RESULT_VECTOR)
    reshape_size = min(x_gt.shape[0], BATCH_SIZE)
    possible_x_trans = possible_x_trans.repeat(reshape_size,1,1)
    possible_x_trans = possible_x_trans.to(device)
    possible_x = torch.transpose(possible_x_trans,1,2)
    results = torch.matmul(possible_x_trans, mat)
    results = torch.matmul(results, possible_x)
    results = torch.diagonal(results, dim1 = 1, dim2 = 2)
    results = torch.topk(results,2,largest=False,dim=1)
    result_value = results.values
    result_angle = results.indices

    a1, a2, a3 = result_angle, result_angle, result_angle
    re_angle = torch.empty((reshape_size, TOTAL_RESULT_VECTOR,1))
    for i in range(reshape_size):
        temp =  possible_x_trans[0,result_angle[i,0]]
        re_angle[i] = temp.reshape((TOTAL_RESULT_VECTOR, 1))
    a1, a2, a3 = torch.split(re_angle, [TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR], dim=1)

    return result_value, a1, a2, a3    

def bruteForceAngle_flexible(mat, x_gt, possible_x_trans, possible_x): 
    results = torch.matmul(possible_x_trans, mat)
    results = torch.matmul(results, possible_x)
    results = torch.diagonal(results, dim1 = 1, dim2 = 2)
    results = torch.topk(results,2,largest=False,dim=1)
    
    reshape_size = x_gt.shape[0]
    
    result_value = results.values
    result_angle = results.indices
    '''
    a1, a2, a3 = result_angle, result_angle, result_angle
    re_angle = torch.empty((reshape_size, TOTAL_RESULT_VECTOR,1))
    for i in range(reshape_size):
        temp =  possible_x_trans[0,result_angle[i,0]]
        re_angle[i] = temp.reshape((TOTAL_RESULT_VECTOR, 1))
    a1, a2, a3 = torch.split(re_angle, [TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR,TOTAL_RESULT_VECTOR], dim=1)
    '''
    temp = possible_x_trans[0,result_angle[0,0]]

    return result_value, temp

possible_x_trans_batch_size = gen_all_binary_vectors(TOTAL_RESULT_VECTOR)
possible_x_trans_batch_size = possible_x_trans_batch_size.repeat(BATCH_SIZE,1,1)
possible_x_trans_batch_size = possible_x_trans_batch_size.to(device)
possible_x_batch_size = torch.transpose(possible_x_trans_batch_size,1,2)

def mat_los_func3(A_mat, x_gt, lamb1, lamb2, network, input):
  A_mat = A_mat.to(device)
  if x_gt.shape[0] == BATCH_SIZE:
    possible_x_trans = possible_x_trans_batch_size
    possible_x = possible_x_batch_size
  else:
    possible_x_trans = gen_all_binary_vectors(TOTAL_RESULT_VECTOR)
    possible_x_trans = possible_x_trans.repeat(x_gt.shape[0],1,1)
    possible_x_trans = possible_x_trans.to(device)
    possible_x = torch.transpose(possible_x_trans,1,2)
  (res_value, _) = bruteForceAngle_flexible(A_mat, x_gt, possible_x_trans, possible_x)
  x_gt.type(torch.FloatTensor)
  reshape_size = min(x_gt.shape[0], BATCH_SIZE)
  x_gt = torch.reshape(x_gt, (reshape_size,TOTAL_RESULT_VECTOR,1))
  x_gt_trans = torch.transpose(x_gt, 1,2)
  result = torch.matmul(x_gt_trans, A_mat)
  result1 = torch.matmul(result, x_gt)
  result = torch.abs(result1 - res_value[:,0])
  A_sum = torch.sum(A_mat,dim=-1)
  A_sum = -0.5 * torch.sum(A_sum, dim = -1)
  #result = torch.flatten(result, start_dim=1) + lamb1 * network.sparse_loss(input) + lamb2 * (A_sum - 1/(2**TOTAL_RESULT_VECTOR) * res_value[:,0])
  result = torch.flatten(result, start_dim=1) + lamb1 * network.sparse_loss(input) - lamb2 * torch.abs(result1 - res_value[:,1])
  result = result.mean()
  return result

torch.autograd.set_detect_anomaly(True)
net1 = Network()
net2 = Network()
net3 = Network()
optm1 = Adam(net1.parameters(), lr=0.0001)
optm2 = Adam(net2.parameters(), lr=0.0001)
optm3 = Adam(net3.parameters(), lr=0.0001)

#device = torch.device("cpu")
net1.to(device)
net2.to(device)
net3.to(device)
#criterion#.to(device)

print("Start Training")
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print("In Epoch: " + str(epoch))
    '''if epoch == 2:
      for g in optm.param_groups:
        g['lr'] = 0.001'''
    criterion = mat_los_func3 # nn.CrossEntropyLoss() #nn.BCELoss()
    running_loss = 0.0
    
    criterion = mat_los_func3 # nn.CrossEntropyLoss() #nn.BCELoss()
    for i, data in enumerate(data_train_snd, 0):


        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optm2.zero_grad()

        # forward + backward + optimize
        outputs = net2(inputs)
        #labels = binary(labels, 9)
        loss = criterion(outputs, labels, 10e-3, 0.001, net2, inputs)
        loss.backward()
        optm2.step()
        gc.collect()
        '''if epoch == 3 and not already_changed:
          for g in optm.param_groups:
            g['lr'] = 0.0001
          already_changed = True'''
        del inputs, labels


        # print statistics
        running_loss += loss.detach().item()
        if i % 20000 == 19999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20000))
            running_loss = 0.0
    
    #if epoch % 10 == 0:      
    #  torch.save(net2, 'snd_network3d' + str(epoch) +'.pt')
    
    criterion = mat_los_func3 # nn.CrossEntropyLoss() #nn.BCELoss()

print('Finished Training')
torch.save(net2, 'our_second_3_1_2.pt')

def angle_difference(P,Q):
  R = torch.transpose(P, -1,-2) @ Q
  RTrace =R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
  RTrace = torch.Tensor([3]) if RTrace > 3 else RTrace
  RTrace = torch.Tensor([-1]) if RTrace < -1 else RTrace
  theta = torch.arccos((RTrace - 1 ) * 0.5)
  return theta

possible_x_trans = gen_all_binary_vectors(TOTAL_RESULT_VECTOR)
possible_x_trans = possible_x_trans.to(device)
possible_x_trans = torch.reshape(possible_x_trans, (1, possible_x_trans.shape[0], possible_x_trans.shape[1]))
possible_x = torch.transpose(possible_x_trans,1,2)

def analysis():
  total = 0
  with torch.no_grad():
    counter = 0
    for index, row in df_data_test.iterrows():
        input_1, input_2, labels = row['Image_1'], row['Image_2'], row['Output']
        size1 = labels.shape[0]
        input = np.matmul(input_2.transpose(), input_1) / len(input_1)

        input = torch.tensor(input)
        new_input = input.flatten()
        new_input = torch.reshape(new_input, (1,new_input.shape[0]))
        new_input = new_input.type(torch.FloatTensor).to(device)
        outputs = net1(new_input)
        #labels = binary(labels, 10)


        labels = torch.tensor(list(labels))
        labels = torch.reshape(labels,(labels.shape[0], 1))
        (_, res_angle1) = bruteForceAngle_flexible(outputs, torch.transpose(labels, 0,1), possible_x_trans, possible_x)
        res_angle1 = res_angle1.type(torch.IntTensor)
        res_angle1 = torch.reshape(res_angle1, (1,TOTAL_RESULT_VECTOR))
        res_angle1 = bin2dec(res_angle1, TOTAL_RESULT_VECTOR)
        res_angle1 = 20 * np.pi / 180 *  res_angle1 / (2**(TOTAL_RESULT_VECTOR)) - 10 * np.pi / 180
        counter_rot_1 = -res_angle1
        mats = calc_rot_mat_from_angle(counter_rot_1.item(),0,0)
  
        new_rot_img = np.matmul(mats, input_2.T)
        new_output = np.matmul(new_rot_img, input_1) / len(input_1)
        new_output = torch.tensor(new_output).flatten()
        new_output = torch.reshape(new_output, (1, new_output.shape[0])).type(torch.FloatTensor).to(device)
        outputs2 = net2(new_output)

        (_,res_angle2) = bruteForceAngle_flexible(outputs2, torch.transpose(labels, 0,1), possible_x_trans, possible_x)
        res_angle2 = res_angle2.type(torch.IntTensor)
        res_angle2 = torch.flatten(res_angle2)
        res_angle2 = bin2dec(res_angle2, TOTAL_RESULT_VECTOR)
        res_angle2 = 10 * np.pi / 180 *  res_angle2 / (2**(TOTAL_RESULT_VECTOR)) - 5 * np.pi / 180
        counter_rot_2 = -res_angle2
        mats_2 = calc_rot_mat_from_angle(0,counter_rot_2,0)

        new_rot_img_2 = np.matmul(mats_2, new_rot_img)
        new_output_2 = np.matmul(new_rot_img_2, input_1) / len(input_1)
        new_output_2 = torch.tensor(new_output_2).flatten()
        new_output_2 = torch.reshape(new_output_2, (1, new_output_2.shape[0])).type(torch.FloatTensor).to(device)
        
        outputs3 = net3(new_output_2)
        (_, res_angle3) = bruteForceAngle_flexible(outputs3, torch.transpose(labels, 0,1), possible_x_trans, possible_x)
        res_angle3 = res_angle3.type(torch.IntTensor)
        res_angle3 = torch.flatten(res_angle3)

        res_angle3 = bin2dec(res_angle3, TOTAL_RESULT_VECTOR)
        res_angle3 = 20 * np.pi / 180 *  res_angle3 / (2**(TOTAL_RESULT_VECTOR)) - 10 * np.pi / 180

        #A_save = conversion
        #X_save = res_angle.cpu()
        total += labels.size(0)
        labels = torch.transpose(labels, 0,1)
        a1, a2, a3 = torch.split(labels, TOTAL_RESULT_VECTOR, dim = 1)
        a1 = torch.reshape(a1,(1,TOTAL_RESULT_VECTOR))
        a2 = torch.reshape(a2,(1,TOTAL_RESULT_VECTOR))
        a3 = torch.reshape(a3,(1,TOTAL_RESULT_VECTOR))
        labels1, labels2, labels3 = bin2dec(a1, TOTAL_RESULT_VECTOR), bin2dec(a2,TOTAL_RESULT_VECTOR), bin2dec(a3,TOTAL_RESULT_VECTOR)

        lables_numpy1 = labels1.cpu().numpy()
        res_angle1 = res_angle1.cpu().numpy()
        lables_numpy2 = labels2.cpu().numpy()
        res_angle2 = res_angle2.cpu().numpy()
        lables_numpy3 = labels3.cpu().numpy()
        res_angle3 = res_angle3.cpu().numpy()

        lables_numpy1 = 20 * np.pi / 180 *  lables_numpy1 / (2**(TOTAL_RESULT_VECTOR))- 10 * np.pi / 180
        lables_numpy2 = 10 * np.pi / 180 *  lables_numpy2 / (2**(TOTAL_RESULT_VECTOR)) - 5 * np.pi / 180
        lables_numpy3 = 20 * np.pi / 180 *  lables_numpy3 / (2**(TOTAL_RESULT_VECTOR))- 10 * np.pi / 180


        P = calc_rot_mat_from_angle(lables_numpy1[0],lables_numpy2[0],lables_numpy3[0])


        Q = calc_rot_mat_from_angle(res_angle1[0],res_angle2,res_angle3)

        theta = angle_difference(P,Q)
        theta = 360 *  theta / (np.pi * 2)
        theta = int(theta)

        histogram[theta] += 1




  print('Accuracy of the network on the test dataset: %d %%' % (
    100 * histogram[0] / total))

correct1, correct2, correct3 = 0,0,0
total = 0
histogram = np.zeros(2**TOTAL_RESULT_VECTOR)
zero1, zero2, zero3 = 0,0,0
counter1, counter2, counter3 = 0,0,0
analysis()

np.save('9layer_histogram2.npy', histogram)

histogram.shape

y_pos = np.arange(2**TOTAL_RESULT_VECTOR)
plt.bar(y_pos[:20], histogram[:20])
plt.show()
t = 0
for i in range(3):
  t += histogram[i]
print('Accurate results within 3 bits: %f ' % (t / (1000)))