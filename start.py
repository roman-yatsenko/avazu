# Initial data transformation

import AvazuDataTransform

path = 'E:\\Avazu\\'  # directory to where you store Avazu data
clTransform = AvazuDataTransform(path)
clTransform.get_Data_Transformed()


# Logistic Regression algorithm

import LogisticRegression

path += 'LR_data\\' # directory to where you store Avazu data
clLR = LogisticRegression(path, 
                          #alpha = 0.0003, #learning rate
                          #n_passes = 6, #number of epoches
                          #poly = True, #to use or not 2nd order polynomial features
                          #wTx=True, #at the end of run wheather to create or not data for MatrixNet to use
                          #adagrad_start = 5) #when to start Adaptive Gradient
clLR.launch()
