#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Question2 on Stochastic gradient descent

#Part A : Data sampling

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# this is how data can be generated
mean =10
std = 1
data_sample = np.random.normal(mean,std,1000)
# print(data_sample)
count, bins, ignored = plt.hist(data_sample, 50, normed=True) 
# print(count)
# print(bins)
# print(ignored)
plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) *
          np.exp( - (bins - mean)**2 / (2 * std**2) ), 
          linewidth=2, color='r') 
plt.show() 


# In[2]:


# Data generated here

# million data points normally distributed
data_size = int(1e6)
m = data_size
x1_list = np.random.normal(3,np.sqrt(4),data_size)
x2_list = np.random.normal(-1,np.sqrt(4),data_size)
noise_list = np.random.normal(0,np.sqrt(2),data_size)
# print(noise_list[:10])

# x1_list = sorted(x1_list,key=float)
# pdf1 = stats.norm.pdf(x1_list,3,2)
# x2_list = sorted(x2_list,key=float)
# pdf2 = stats.norm.pdf(x2_list,-1,2)
# noise_list = sorted(noise_list,key=float)
# pdf3 = stats.norm.pdf(noise_list,0,np.sqrt(2))


# plt.plot(x1_list,pdf1)
# plt.plot(x2_list,pdf2)
# plt.plot(noise_list,pdf3)


# In[3]:


# given theta_vector and x1 and x2 vectors , lets try to see the distribution of yi with error vector 

# 3x1
theta = np.array([[ 3,1,2 ]])
theta = theta.T

# mx1
epsilon = np.array([noise_list])
epsilon = epsilon.T           

# method to add rows in matrix
#theta = np.vstack((theta,np.array([[4]])))
#theta.shape


# mx3
x_vector = np.array([[ 1,x1_list[0], x2_list[0] ]])

# takes a lot of time to append for million data points
#--------------------------------------
# for i in range(len(x1_list)-1):
#     new_sample = np.array([[ 1 , x1_list[i+1], x2_list[i+1]    ]])
#     x_vector = np.vstack( ( x_vector , new_sample ) )

# print(x_vector.shape)    

# y_vector = np.matmul(x_vector,theta) + epsilon
#-----------------------------------------------------



# approach 2, still takes time but lesser
# y_list =[]
# for i in range(len(x1_list)):
#     y_list.append(  theta[0][0]*1 + theta[1][0]*x1_list[i] + theta[2][0]*x2_list[i] + epsilon[i][0]   )

# y_vector = np.array([y_list])
# y_vector = y_vector.T
#--------------------------------------------------------------------


#Approach3 : column appending


x_vector = np.ones((int(1e6),1))

x1_vector = np.array([x1_list])
x1_vector = x1_vector.T

x2_vector = np.array([x2_list])
x2_vector = x2_vector.T

x_vector = np.append( x_vector, x1_vector,axis = 1 )

x_vector = np.append( x_vector, x2_vector,axis = 1 )
y_vector = np.matmul(x_vector,theta) + epsilon




# print(y_vector[:10])


mean = np.mean(y_vector)
std = np.std(y_vector)

count, bins, ignored = plt.hist(y_vector, 50, normed=True) 
# print(count)
# print(bins)
# print(ignored)
plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) *
          np.exp( - (bins - mean)**2 / (2 * std**2) ), 
          linewidth=2, color='r') 
plt.show() 
# it also seems to follow normal distribution with mean (theta.T*X)


# In[4]:


def grad_J_theta(x_vec,y_vec,theta):  # X : mx3   ,  Y : mx1   , theta : 3x1
    
    abs_error =  y_vec - (x_vec @ theta)
    #print(abs_error)
    delta = abs_error.T @ x_vec     # matrix multiplication summation happening here
    delta = delta.T
    cost = (1.0/(2*len(x_vec)))* ( abs_error.T @ abs_error  )
    cost = cost[0][0]
    return delta/len(x_vec)
    


# In[5]:


def cost(x_vec,y_vec,theta):
    abs_error =  y_vec - (x_vec @ theta)
    #cost = (1.0/(2*len(x_vec)))* ( abs_error.T @ abs_error  )
    cost = (1.0/2)* ( abs_error.T @ abs_error  )
    return cost[0][0]/len(x_vec)


# In[6]:


#------------------------ STOCHASTIC GRADIENT DESCENT--------------------------------------------------

# x_vector  : mx3
# y_vector  : mx1

#Initialize theta
theta = np.zeros(3).reshape(3,1)

#batch_size = 1000



#Shuffle the data
x_y_vector = np.append(x_vector,y_vector,axis = 1)
np.random.shuffle(x_y_vector)          # taking time



    


# In[ ]:





# In[ ]:





# In[7]:


def min_max_diff(history):   # returns min and max vectors 3x1
    temp =  history[0]
    for i in range( 1,len(history) ):
        temp = np.append(temp,history[i],axis=1)
    #print(temp)
    min = np.amin(temp,axis=1).T
    max = np.amax(temp,axis=1).T
    #print(max - min < 2)
    return min,max


# In[8]:




learning_rate = 0.001
converged = False 
prev_cost = 99999
curr_cost = 0
avg_over = 100000
batch_size = 10000   # Note re initialized
examples = 0
total_cost = 0
total_batches = m//batch_size     # to handle boundry case

update_limit = 20000
update=0

history0=[]
history1=[]
history2=[]

plot_theta_0=[]
plot_theta_1=[]
plot_theta_2=[]

error = 1e-4
#history.append(theta)


while converged == False :
  
    for b in range(total_batches):
        
        xb = x_vector[ b*batch_size : (b+1)*batch_size ]
        yb = y_vector[ b*batch_size : (b+1)*batch_size ]
        
        #print(len(xb),b)
        total_cost += cost(xb,yb,theta)
        #print(total_cost)
        examples += batch_size
        
        theta += learning_rate* grad_J_theta(xb,yb,theta)
        #print(theta.T)
        update+=1
        
        plot_theta_0.append(theta[0][0])
        plot_theta_1.append(theta[1][0])
        plot_theta_2.append(theta[2][0])
        
        
        #history.append( theta )
        #print(len(history))
        if len(history0)<=2:
            history0.append(theta[0][0])
            history1.append(theta[1][0])
            history2.append(theta[2][0])
            
        else:
            history0.pop(0)
            history1.pop(0)
            history2.pop(0)
            
            history0.append(theta[0][0])
            history1.append(theta[1][0])
            history2.append(theta[2][0])
            
            if abs( max(history0)-min(history0) ) < error  and abs( max(history1)-min(history1) ) < error  and abs( max(history2)-min(history2) ) < error :
                converged = True
                print("coverged",theta.T)
                print(abs( max(history0)-min(history0) ),abs( max(history1)-min(history1) ),abs( max(history2)-min(history2) ))
                break
        
        
            
            

        
        
        if examples%avg_over == 0 :  # reduces print statements
            
            print(theta.T,print(abs( max(history0)-min(history0) ),abs( max(history1)-min(history1) ),abs( max(history2)-min(history2) )))
            
#             curr_cost = total_cost/( avg_over/batch_size )        # total cost / no. of itrs
#             #print(theta.T,curr_cost,abs(prev_cost - curr_cost),update,len(history))
#             if abs(prev_cost - curr_cost) < 1e-3 :
#                 #converged = True
#                 break
#             else:
#                 prev_cost = curr_cost
#                 curr_cost = 0
#                 total_cost = 0
#                 examples = 0
            
        




# In[9]:


print(history0)
print(update)


# In[10]:



print("Final obtained theta :",theta)
cost1  = cost(x_vector,y_vector,theta)
print(cost1)

# theta = np.array([[ 3,1,2 ]])
# theta = theta.T
# cost1  = cost(x_vector,y_vector,theta)
# print(cost1)
# print(x_vector.shape)


# In[11]:


test = open("q2test.csv","r")
x1_test=[]
x2_test=[]
y_test=[]

for i,line in enumerate(test):
    if(i==0):
        continue
    #print(line.split(","))
    x1_test.append(float(line.split(",")[0]))
    x2_test.append(float(line.split(",")[1]))
    y_test.append(float(line.split(",")[2].split("\n")[0]))
#print(y_test)    


# In[12]:



x_vector = np.ones((len(x1_test),1))
x1_vector = np.array([x1_test])
x1_vector = x1_vector.T

x2_vector = np.array([x2_test])
x2_vector = x2_vector.T

x_vector = np.append( x_vector, x1_vector,axis = 1 )

x_vector = np.append( x_vector, x2_vector,axis = 1 )
y_vector = np.array([y_test]).T

print("learnt theta cost : ",cost(x_vector,y_vector,theta))
print(" Cost with theta\[3,2,1] :",cost(x_vector,y_vector,np.array([[3,1,2]]).T))


# In[ ]:





# In[13]:


from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot( plot_theta_0 , plot_theta_1, plot_theta_2, color="blue" )
ax.set_xlabel('Theta_0')
ax.set_ylabel('Theta_1')
ax.set_zlabel('Theta_2')
plt.title("Batch size : "+str(batch_size))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




