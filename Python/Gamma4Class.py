import scipy
from scipy import io

import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join

mypath = 'ABC/A_Gamma/'


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
#print(onlyfiles[1])
no_of_files = len(onlyfiles)
Matrix_C = []
Shape_of_Sample =[]


#print('after')
for i in range(1, 87):                     ##306 first sample is temp
    temp = mypath+onlyfiles[i]
    #print(temp)
    mat1 = scipy.io.loadmat(temp)
    Coeffs = mat1['Matrix'];
    #print(Coeffs.shape)
    Coeffs = np.ascontiguousarray(Coeffs.T);
    Temp = np.size(Coeffs,0)
    Shape_of_Sample.append(Temp)
    Matrix_C.append(Coeffs)


#print(Matrix_C[0].shape)
#print('hello')
mypath2 = 'ABC/Class2.mat'
mat2 = scipy.io.loadmat(mypath2)
Class_Training = mat2['Class_Training']
Class_Testing= mat2['Class_Testing']

mypath3 = 'ABC/B_Gamma/'
onlyfiles3 = [f3 for f3 in listdir(mypath3) if isfile(join(mypath3,f3))]
for j in range(1, 136):                     ##306 first sample is temp
    temp3 = mypath3+onlyfiles3[j]
    #print(temp3)
    mat3 = scipy.io.loadmat(temp3)
    Coeffs3 = mat3['Matrix'];
    #print(Coeffs.shape)
    Coeffs3 = np.ascontiguousarray(Coeffs3.T);
    Temp3 = np.size(Coeffs3,0)
    Shape_of_Sample.append(Temp3)
    Matrix_C.append(Coeffs3)

mypath4 = 'ABC/C_Gamma/'
onlyfiles4 = [f4 for f4 in listdir(mypath4) if isfile(join(mypath4,f4))]
for k in range(1, 145):                     ##306 first sample is temp
    temp4 = mypath4+onlyfiles4[k]
    #print(temp4)
    mat4 = scipy.io.loadmat(temp4)
    Coeffs4 = mat4['Matrix'];
    #print(Coeffs.shape)
    Coeffs4 = np.ascontiguousarray(Coeffs4.T);
    Temp4 = np.size(Coeffs4,0)
    Shape_of_Sample.append(Temp4)
    Matrix_C.append(Coeffs4)

mat10 = scipy.io.loadmat('ABC/OfficeG.mat')
Coeffs10 = mat10['Matrix']
Coeffs10 = np.ascontiguousarray(Coeffs10.T);
Temp10 = np.size(Coeffs10,0)
Shape_of_Sample.append(Temp10)
Matrix_C.append(Coeffs10)
#print(np.size(Shape_of_Sample))
#print(i)
Mean_C = []
#print(Matrix_C[0].shape)
for m in range(0, 366):
    #print('value of i is ', i)
    Temp2 = Shape_of_Sample[m]
    #sum = sum +Temp2
    for n in range(0, Temp2):
        #print('value of j is ',j)
        Temp3 = np.mean(Matrix_C[m], axis = 0)
        Mean_C.append(Temp3)

Mean_C_2 = np.mean(Mean_C,axis=0)
print(Mean_C_2)

#print('before is ',Matrix_C[:])

for i in range(0, 366):
    Matrix_C[i] = Matrix_C[i] - Mean_C_2


Training = np.concatenate(Matrix_C[:], 0)
Testing = []
Shape_of_Sample2 =[]
mypath6 = 'ABC/Testing_Gamma/'
onlyfiles6 = [f6 for f6 in listdir(mypath6) if isfile(join(mypath6,f6))]
for o in range(1, 29):                     ##306 first sample is temp
    temp6 = mypath6+onlyfiles6[o]
    #print(temp6)
    mat6= scipy.io.loadmat(temp6)
    Coeffs6= mat6['Matrix'];
    #print(Coeffs.shape)
    Coeffs6 = np.ascontiguousarray(Coeffs6.T);
    Temp6 = np.size(Coeffs6,0)
    Shape_of_Sample2.append(Temp6)
    Testing.append(Coeffs6)

mypath7 = 'ABC/Testing_Gamma2/'
onlyfiles7 = [f7 for f7 in listdir(mypath7) if isfile(join(mypath7,f7))]
for p in range(0, 16):                     ##306 first sample is temp
    temp7 = mypath7+onlyfiles7[p]
    #print(temp7)
    mat7= scipy.io.loadmat(temp7)
    Coeffs7= mat7['Matrix'];
    #print(Coeffs.shape)
    # print(Coeffs.shape)
    Coeffs7 = np.ascontiguousarray(Coeffs7.T);
    Temp7 = np.size(Coeffs7, 0)
    Shape_of_Sample2.append(Temp7)
    Testing.append(Coeffs7)

mat13 = scipy.io.loadmat('ABC/OfficeTG.mat')
Coeffs13 = mat13['Matrix']
Coeffs13 = np.ascontiguousarray(Coeffs13.T);
Temp13 = np.size(Coeffs13, 0)
Shape_of_Sample2.append(Temp13)
Testing.append(Coeffs13)

print(np.size(Shape_of_Sample2))

for i in range(0, 45):
    Testing[i] = Testing[i] - Mean_C_2

Testing = np.concatenate(Testing[:], 0)

