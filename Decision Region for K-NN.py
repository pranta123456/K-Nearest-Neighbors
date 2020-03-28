#!/usr/bin/env python
# coding: utf-8

# Decision Boundary for K-NN:

# In[1]:


#Importing a bunch of libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions


# Method-1:

# In[8]:


#Define some core function
def knn_comparison(data,k):
    data=np.array(data)
    x=data[:,:-1]
    y=data[:,-1].astype(int)
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x,y)
    plot_decision_regions(x,y,clf=neigh,legend=2)     #Plotting decision regions
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("K-NN with K="+str(k))
    plt.show()


# In[9]:


#U-Shape Data
data1=pd.read_csv("ushape.csv")
for i in [1,5,15,30]:
    knn_comparison(data1,i)


# In[4]:


#Concentric_circle Data
data2=pd.read_csv("concertriccir2.csv")
for i in [1,5,15,30]:
    knn_comparison(data2,i)


# In[5]:


#XOR Data
data3=pd.read_csv("xor.csv")
for i in [1,5,15,30]:
    knn_comparison(data3,i)


# In[6]:


#Linearly Separable Data
data4=pd.read_csv("linearsep.csv")
for i in [1,5,15,30]:
    knn_comparison(data4,i)


# In[7]:


#Outlier Data
data5=pd.read_csv("outlier.csv")
for i in [1,5,15,30]:
    knn_comparison(data5,i)


# Method-2:

# In[12]:


#Import one imp library
from matplotlib.colors import ListedColormap


# In[39]:


#Define Core function
def knn_comparison_1(data,k):
    data_0=np.delete(data,0,axis=0)
    X=data_0[:, :2]
    y=data_0[:,2]
    h=.02 #Grid cell size
    cmap_light=ListedColormap(["#FFAAAA","#AAAAFF"]) #for smooth region two hexcolor
    cmap_bold=ListedColormap(["#FF0000","#0000FF"])  #for points two hexcolor
    #The core classifier: K-NN
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X,y)
    #Specify the range of meshgrid
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    #We create a meshgrid from (x_min,y_min) to (x_max,y_max) with step h=.02
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    #Predict the cls label of each point in the grid
    Z=neigh.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    #pcolormesh will color the (xx,yy) grid according to the value
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
    #scatter plot with given points
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold)
    #define scale on both axis
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    #set the title
    plt.title("K-NN with K="+str(k))
    plt.show()
    
    


# In[40]:


#Ushape Data
data_1=np.genfromtxt("ushape.csv",delimiter=",")
for i in [1,5,15,30]:
    knn_comparison_1(data_1,i)


# In[11]:


#Concentric_circle Data
data_2=np.genfromtxt("concertriccir2.csv",delimiter=",")
for i in [1,5,15,30]:
    knn_comparison_1(data_2,i)


# In[12]:


#XOR Data
data_3=np.genfromtxt("xor.csv",delimiter=",")
for i in [1,5,15,30]:
    knn_comparison_1(data_3,i)   


# In[13]:


#Linearly Separable Data
data_4=np.genfromtxt("linearsep.csv",delimiter=",")
for i in [1,5,15,30]:
    knn_comparison_1(data_4,i)


# In[14]:


#Outlier Data
data_5=np.genfromtxt("outlier.csv",delimiter=",")
for i in [1,5,15]:
    knn_comparison_1(data_5,i)

