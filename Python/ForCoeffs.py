import scipy
from scipy import io

import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join

mypath = 'Using only coeff/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
#print(onlyfiles[1])
no_of_files = len(onlyfiles)
Matrix_C = []
Shape_of_Sample =[]

mat2 = scipy.io.loadmat('labels.mat')
Class_Training = mat2['Labels'];
for i in range(1, 306):                     ##306 first sample is temp
    temp = mypath+onlyfiles[i]
    #print(temp)
    mat1 = scipy.io.loadmat(temp)
    Coeffs = mat1['Matrix_Coeffs'];
    Coeffs = np.ascontiguousarray(Coeffs.T);
    Temp = np.size(Coeffs,0)
    Shape_of_Sample.append(Temp)
    Matrix_C.append(Coeffs)


Mean_C = []
#find mean of every individual list in the parent list
for i in range(0, 305):
    Temp2 = Shape_of_Sample[i]
    for j in range(0, Temp2):
        Temp3 = np.mean(Matrix_C[i], axis = 0)
        Mean_C.append(Temp3)

Mean_C_2 = np.mean(Mean_C,axis=0)

for i in range(0, 305):
    Matrix_C[i] = Matrix_C[i] - Mean_C_2


Training = np.concatenate(Matrix_C[:], 0)
