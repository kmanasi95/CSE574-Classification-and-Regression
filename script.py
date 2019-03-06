import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    ## Outputs
    
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    

    # IMPLEMENT THIS METHOD     
    data_map = {}
    means = []
    for i in range(X.shape[0]):
        key = int(y[i])
        try:
            data_map[key].append(X[i])
        except KeyError:
            data_map[key] = [X[i]]      
    for key, value in data_map.items():
        data_map[key] = np.asarray(data_map[key])
        means.append(np.mean(data_map[key], axis=0))
    means = np.asarray(means).T 
    
    covmats = np.cov(X.T)
    return means,covmats


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    data_map = {}
    means = []
    covmats = []
    for i in range(X.shape[0]):
        key = int(y[i])
        try:
            data_map[key].append(X[i])
        except KeyError:
            data_map[key] = [X[i]]      
    for key, value in data_map.items():
        data_map[key] = np.asarray(data_map[key])
        means.append(np.mean(data_map[key], axis=0))
        covmats.append(np.cov(data_map[key].T))
    means = np.asarray(means).T 
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    ypred = []
    acc = 0
    for j in range(Xtest.shape[0]):
        result = []
        for i in range(means.shape[1]):
            exponent = ((np.matmul(np.matmul((Xtest[j] - means.T[i]).T, np.linalg.inv(covmat)), (Xtest[j] - means.T[i]))) / 2)
            result.append(exponent)
            
        ypred.append(np.argmin(result) +1)
        
        if ypred[j] == ytest[j]:
            acc+=1
        
    print(ypred)
    acc = acc/Xtest.shape[0]
    ypred = np.asarray(ypred).reshape(ytest.shape[0],1)
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    ypred = []
    acc = 0
    for j in range(Xtest.shape[0]):
        result = []
        for i in range(means.shape[1]):
            determinant = np.linalg.det(covmats[i])
            exponent = np.exp(-((np.matmul(np.matmul((Xtest[j] - means.T[i]).T, np.linalg.inv(covmats[i])), (Xtest[j] - means.T[i]))) / 2))
            constant = 1 / (np.sqrt(2 * np.pi * (determinant ** 2)))
            result.append(constant * exponent)
            
        ypred.append(np.argmax(result) + 1)
        
        if ypred[j] == ytest[j]:
            acc+=1
        
    acc = acc/Xtest.shape[0]
    ypred = np.asarray(ypred).reshape(ytest.shape[0],1)
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD 
    return np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))                                               

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1
    # seting derivative d(J(w))/d(w) = 0, we get -2X'(y-Xw) + 2.lambd.w = 0
    # solving for w => w = (X'X + lambd.I)'*(X'y)
    dimR, dimC = X.shape
    expression1 = np.matmul(X.T, X) + lambd * np.identity(dimC)
    inv_expression1 = np.linalg.inv(expression1)
    expression2 = np.matmul(X.T, y)
    weight = np.matmul(inv_expression1, expression2)                                                   
    return weight

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    return (np.sum((ytest - np.dot(Xtest,w)) ** 2)) / Xtest.shape[0]

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    w = w.flatten() 
    w = w.reshape(w.shape[0],1)
    diff = y - np.matmul(X, w)
    error = (np.dot(diff.T, diff) + (lambd*np.dot(w.T, w))) * 0.5
    error_grad = -2*(np.matmul(X.T, diff)) + 2*lambd*w
    error_grad = error_grad.reshape(error_grad.shape[0])
    return error, error_grad
    
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    
    #Initialize output array Xp
    Xp = np.zeros((x.shape[0], p+1))
    
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xp[i][j] = x[i] ** j
            
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
print("LDA mean",means)
ldaacc, ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')
plt.show()


# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
    
#print(mses3_train)
#print(mses3)
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[mses3.tolist().index(min(mses3))] # lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    
#print("Train error when lambda is 0 and optimal : ",mses5_train)
#print("Test error when lambda is 0 and optimal : ",mses5) 
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()