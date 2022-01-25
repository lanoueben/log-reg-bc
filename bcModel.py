#Author: Benjamin Lanoue
#logistic regression
#For pandas 2D arrays, X[row][column]
import numpy as np
import pandas as pd
import random as r
import math
import matplotlib.pyplot as plt


#This assumes that the data has been proprocessed as well as
#removing the y output column
#input: a pandas dataframe with only independent variables
#output:A normalized pandas dataframe with only ind vars
def normalize(df):
    
    rowNum = len(df.index)
    colNum = len(df.columns)
    
    for i in range(colNum):
        
        #find min(), max(), and mean
        minX = df.iloc[0,i]
        maxX = df.iloc[0,i]
        sumX = 0
        for j in range(rowNum):
            element = df.iloc[j,i]
            if element > maxX:
                maxX = element
            if element < minX:
                minX = element
            sumX =+ element 
        #end min,max,sum, 
        mean = sumX/rowNum
        
        #mean normalization on the data
        for j in range(rowNum):
            element = df.iloc[j,i]
            deviation = maxX-minX
            df.iloc[j,i] = (element-mean)/deviation
    
    return df

#input: X_i represents a preprosessed 2d pandas object row, and 
#       params is an array of thetas, that include +1  (params[0])
#output:The hypothesis of the currenty X_i (A value)
#       If the hypothesis value is <.5, then the prediction
#       is 0, if hypothesis value >=.5. prediction is 1
def calculate_h_x(X_i, params, row):
    prediction = params[0]
    colNum = len(X_i.columns)
    
    for i in range(colNum):
        prediction = prediction + (X_i.iloc[row][i+2]* params[i+1])
    #end for i
    
    hypothesis = 1/(1+(math.e**(-1*prediction)))
    
    
    return hypothesis

#Calculate the cost of the current parameter
#input: X represents a preprocessed 2D pandas object, params
#       is an array of thetas, that include +1 (params[0])
#       y represents a preprocessed array of dependent variables
#       corresponding to X
#output:return a cost value for row of X&y,based on row
def calculate_loss(X,y,params,row):
    h_x = calculate_h_x(X,params,row)
    cost = -y[row]*math.log(h_x) - (1-y[row])*math.log(1-h_x)
        
    return cost

#h_x is an array the same size as y that has a prediciton
def calculate_cost(h_x, y):
    rows = len(y)
    cost = 0    
    for i in range(rows):
        cost = cost+ (-y[i]*math.log(h_x[i]) - (1-y[i])*math.log(1-h_x[i]))/rows
    
    return cost

#input: X represents a preprocessed 2D pandas object
#       containing dependent variables
#       y is an array of corresponding dependent variables
#       to X
#       alpha is a learning rate, and T is the iterations
#output:an array of thetas that have been minimized
#       graph showing the cost function
def gradient_descent(X,y,alpha,T):
    
    
    #costValues will store a list of costs. Each cost will record   
    #an iteration of the current thetas cost.
    #xcostValues will store the iteration identities
    costValues = []
    xcostValues = []
    colNum = len(X.columns)
    rowNum = len(y)

#First, establish thetas values random value bw 0.01-1.0
#This will be used to calculate the hypothesis
    theta = []
    for i in range(colNum + 1):
        theta.append(r.random())
    #end for i
    
    
#Next, we create a loop to indicate the iterations for cost minimization


#i will represent the iteation
    for i in range(T):

        #this will be the size of the amount of rows in the data, and
        #will hold a temporary hypothesis for each row
        tempHypothesis = []
        
        
        #create a loop that will iterate through all of the rows
        #j will stand for an instance of the data 
        for j in range(rowNum):
            tempHypothesis.append( calculate_h_x(X, theta, j) )
        #end for j
        
        
        #simultaneously update/calculate theta. First iterate through all
        #columns, then iterate through all the rows.
        
        for j in range(colNum+1):
            tempTheta = 0
            for k in range(rowNum):
                if (j==0):
                    tempTheta = tempTheta + (tempHypothesis[k] - y[k])/rowNum
                else:
                    tempTheta = tempTheta + (tempHypothesis[k]- y[k])*X.iloc[k][j+1]/rowNum
            #end for k
            
            theta[j] = theta[j] - alpha*tempTheta
            
        #end for j
        
        tempHypothesis = []
        for j in range(rowNum):
            tempHypothesis.append( calculate_h_x(X, theta, j) )
        
        #calclate cost values
        xcostValues.append(i)
        tempCost = calculate_cost(tempHypothesis, y)
        costValues.append(tempCost)
    #end for i
    
    
    plt.plot(xcostValues,costValues)
    
    return theta




#-----------------------------------------------------------------------------

wdbc = pd.read_csv("wdbc.data",header=None)
#wdbc.iloc[row,column]


#this counts the amount of rows
xnum = len(wdbc.index)

#This will drop column 0, which is an identifier column.
#axis=1 maakes it so it drops a column, having it = 0 will make it drop a row
wdbc = wdbc.drop(wdbc.columns[[0]],axis=1)

y = []
#Change M to 1, and B to 0 so nomralize() can classify the values.
#Hold these values in the y[] array, which allows the
#corresponding column to be dropped, leaving only
#independent variables in the wdbc df
for i in range(xnum):
    if "M" == wdbc.iloc[i,0]:
        y.append(1)
    else:
        y.append(0)
#End for i loop

x = wdbc.drop(wdbc.columns[[0]],axis=1)

#This counts the amount of attributes, starts at 1 instead of 0
columnNum = len(x.columns)

print(x.iloc[0,0])
#wdbc.iloc[0,0] = 842301
#print(wdbc.iloc[0,0]); 



def main():
    normalize(x)
    correct = 0
    temp = [-0.13019641397227097, 0.5013903243371093, 0.15014756131861975, -0.10826523921500485, 0.08130600218304652, -0.29661766783335863, 0.6927028468698342, 0.8976642826559935, 0.7732330805110962, -0.6610710435969134, -0.6703958396248579, 0.576301472703391, 0.4914571911941294, 0.5833924703497689, 0.06947340525390948, 0.029887570007273767, 0.43752635923260774, 0.6024059034515972, -0.09959638870635232, 0.19253103921788406, 0.5122467190125878, -0.011269683036978702, -0.48080261639166927, 0.27948527274891577, 0.3807162814984631, -0.7232723696495073, 0.32821835829468715, 0.8669005234144801, 0.44344132634032746, 0.17144938508860846, 0.21617069309989176]
    for i in range(xnum):
        h = calculate_h_x(x, temp, i)
        if h>=.5:
            h=1
        else:
            h=0
        if y[i] == h:
            correct+= 1
    #end for i
    print(correct/xnum)
    pass
    #the = gradient_descent(x, y, .0005, 10000)
    #print(the)
    
main()



def createTheta(cNum):
    a = []
    for i in range(cNum):
        a.append(i)
    
    return a

#test  calculate_h_x(X_i, params, row)
def test1():
    a = createTheta(columnNum+1)
    
    temp = calculate_h_x(x,a,1)
    print(temp)
    pass


#test calculate_loss(X,y,params,row)
# cost = -y[row]*math.log10(h_x) - (1-y[row])*math.log10(1-h_x)
def test2():
    
    oneCount = 0
    zeroCount = 0
    
    for i in y:
        if i == 1:
            oneCount += 1
        else:
            zeroCount +=1
            
    print(zeroCount, "zero values")
    print(oneCount, "one values")
    
    a = createTheta(columnNum+1)
    
    temph = calculate_h_x(x, a, 0)
    print(temph)
    print(-y[0]*math.log(temph))
    
    
    pass

                                                                            
#gradient_descent(X,y,alpha,T)
#Theres a lot to test here
def test3():
    
    pass

"""
def accuracy():
    correct = 0
    temp = []
    for i in range(xnum):
        h = calculate_h_x(x, temp, i)
        if h>=.5:
            h=1
        else:
            h=0
        if y[i] == h:
            correct+= 1
    #end for i
    print(correct/xnum)
    pass

[-0.13019641397227097, 0.5013903243371093, 0.15014756131861975, -0.10826523921500485, 0.08130600218304652, -0.29661766783335863, 0.6927028468698342, 0.8976642826559935, 0.7732330805110962, -0.6610710435969134, -0.6703958396248579, 0.576301472703391, 0.4914571911941294, 0.5833924703497689, 0.06947340525390948, 0.029887570007273767, 0.43752635923260774, 0.6024059034515972, -0.09959638870635232, 0.19253103921788406, 0.5122467190125878, -0.011269683036978702, -0.48080261639166927, 0.27948527274891577, 0.3807162814984631, -0.7232723696495073, 0.32821835829468715, 0.8669005234144801, 0.44344132634032746, 0.17144938508860846, 0.21617069309989176]
"""


















