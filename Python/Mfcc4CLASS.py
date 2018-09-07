import scipy
from scipy import io

import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join

mypath = 'ABC/A_MFCC/Coeff/'
sum = 0

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
#print(onlyfiles[1])
no_of_files = len(onlyfiles)
Matrix_C = []
Shape_of_Sample =[]


#print('after')
for i in range(0, 86):                     ##306 first sample is temp
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
#print('after')
mypath2 = 'ABC/Class3.mat'
mat2 = scipy.io.loadmat(mypath2)
Class_Training = mat2['Class_Training']
Class_Testing= mat2['Class_Testing']


mypath3 = 'ABC/B_MFCC/Coeff/'
onlyfiles3 = [f3 for f3 in listdir(mypath3) if isfile(join(mypath3,f3))]
for j in range(0, 135):                     ##306 first sample is temp
    temp3 = mypath3+onlyfiles3[j]
    #print(temp3)
    mat3 = scipy.io.loadmat(temp3)
    Coeffs3 = mat3['Matrix'];
    #print(Coeffs.shape)
    Coeffs3 = np.ascontiguousarray(Coeffs3.T);
    Temp3 = np.size(Coeffs3,0)
    Shape_of_Sample.append(Temp3)
    Matrix_C.append(Coeffs3)

mypath4 = 'ABC/C_MFCC/Coeff/'
onlyfiles4 = [f4 for f4 in listdir(mypath4) if isfile(join(mypath4,f4))]
for k in range(0, 144):                     ##306 first sample is temp
    temp4 = mypath4+onlyfiles4[k]
    #print(temp4)
    mat4 = scipy.io.loadmat(temp4)
    Coeffs4 = mat4['Matrix'];
    #print(Coeffs.shape)
    Coeffs4 = np.ascontiguousarray(Coeffs4.T);
    Temp4 = np.size(Coeffs4,0)
    Shape_of_Sample.append(Temp4)
    Matrix_C.append(Coeffs4)

mat10 = scipy.io.loadmat('ABC/OfficeM3.mat')
Coeffs10 = mat10['Matrix']
Coeffs10 = np.ascontiguousarray(Coeffs10.T);
Temp10 = np.size(Coeffs10,0)
Shape_of_Sample.append(Temp10)
Matrix_C.append(Coeffs10)

print(Matrix_C[363].shape)
print(Matrix_C[364].shape)
print(Matrix_C[365].shape)
#print(Matrix_C[366].shape)

#print(i)

Mean_C = []

for m in range(0, 366):
    #print('value of i is ', i)
    Temp2 = Shape_of_Sample[m]
    #sum = sum +Temp2
    for n in range(0, Temp2):
        #print('value of j is ',m)
        Temp3 = np.mean(Matrix_C[m], axis = 0)
        Mean_C.append(Temp3)

Mean_C_2 = np.mean(Mean_C,axis=0)

for i in range(0, 366):
    Matrix_C[i] = Matrix_C[i] - Mean_C_2



Training = np.concatenate(Matrix_C[:], 0)
Testing = []
Shape_of_Sample2 =[]
mypath6 = 'ABC/Testing_MFCC/'
onlyfiles6 = [f6 for f6 in listdir(mypath6) if isfile(join(mypath6,f6))]
for o in range(0, 28):                     ##306 first sample is temp
    temp6 = mypath6+onlyfiles6[o]
    #print(temp6)
    mat6= scipy.io.loadmat(temp6)
    Coeffs6= mat6['Matrix'];
    #print(Coeffs.shape)
    Coeffs6 = np.ascontiguousarray(Coeffs6.T);
    Temp6 = np.size(Coeffs6,0)
    Shape_of_Sample2.append(Temp6)
    Testing.append(Coeffs6)

mypath7 = 'ABC/Testing_MFCC2/'
onlyfiles7 = [f7 for f7 in listdir(mypath7) if isfile(join(mypath7,f7))]
for p in range(0, 16):                     ##306 first sample is temp
    temp7 = mypath7+onlyfiles7[p]
    #print(temp7)
    mat7= scipy.io.loadmat(temp7)
    Coeffs7= mat7['Matrix'];
    #print(Coeffs.shape)
    Coeffs7 = np.ascontiguousarray(Coeffs7.T);
    Temp7 = np.size(Coeffs7,0)
    Shape_of_Sample2.append(Temp7)
    Testing.append(Coeffs7)
#print('before')
#print(np.size(Shape_of_Sample2))
#print(Training[0].shape)
#print(Testing[28].shape)

mat13 = scipy.io.loadmat('ABC/OfficeTM2.mat')
Coeffs13 = mat13['Matrix']
Coeffs13 = np.ascontiguousarray(Coeffs13.T);
Temp13 = np.size(Coeffs13,0)
Shape_of_Sample2.append(Temp13)
Testing.append(Coeffs13)

for i in range(0, 45):
    Testing[i] = Testing[i] - Mean_C_2
    ##print(i)

Testing = np.concatenate(Testing[:], 0)


print(Testing.shape)

'''
print(Training.shape)
print('New')
print(Testing.shape)

mypath2 = '2 seconds data test/'

onlyfiles2 = [f2 for f2 in listdir(mypath2) if isfile(join(mypath2,f2))]
no_of_files2 = len(onlyfiles2)

Testing = []
Shape_of_Sample2 =[]
#print('in e')
for i in range(1, 31):                     ##306 first sample is temp
    temp2 = mypath2+onlyfiles2[i]
    #print(temp2)
    mat2 = scipy.io.loadmat(temp2)
    Coeffs2 = mat2['Matrix'];
    Coeffs2 = np.ascontiguousarray(Coeffs2.T);
    Temp2 = np.size(Coeffs2,0)
    Shape_of_Sample2.append(Temp2)
    Testing.append(Coeffs2)

#print(Testing[0].shape)
for i in range(0, 30):
    Testing[i] = Testing[i] - Mean_C_2

Testing = np.concatenate(Testing[:], 0)
print(Training.shape)
print('New')
print(Testing.shape)

'''


