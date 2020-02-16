#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import time


# In[2]:


linearX = open("linearX.csv","r")
X_list=[]
for line in linearX:
    #print(float(line.split("\n")[0]))    
    X_list.append(float(line.split("\n")[0]))
#print(X_list)

linearY = open("linearY.csv","r")
Y_list=[]
for line in linearY:
    #print(float(line.split("\n")[0]))    
    Y_list.append(float(line.split("\n")[0]))
#print(Y_list)


# In[3]:


#Normalizing X feature

sum = 0
for i in range(len(X_list)):
    sum+= X_list[i]
mean = sum/len(X_list)    
#print(mean)
temp = 0
for i in range(len(X_list)):
    temp+=(X_list[i]-mean)**2
variance = temp/len(X_list)    
#print(variance)
std = math.sqrt(variance)
#print(std)

for i in range(len(X_list)):
    X_list[i]=(X_list[i] - mean)/std

#print(X_list)


# In[28]:


plt.scatter(X_list,Y_list,color='blue')


# In[5]:


x_vec = np.array([X_list])
x_vec = np.transpose(x_vec)
#print(x_vec)
print(x_vec[1])


# In[6]:


#Initializing theta and x vector
theta = np.array([[0.,0.]]) 
theta = np.transpose(theta)

x_i = np.array([[1,x_vec[0]]])
x_i = np.transpose(x_i)
print(theta.shape,x_i.shape)


# In[7]:


def hypo_xi(theta,x_vec):
    result = np.dot(theta.T,x_vec)
    #print(result[0][0])
    return result[0][0]


# In[8]:


hypo_xi(theta,x_i)


# In[9]:


def grad_J_theta(X_list,Y_list,theta):
    result = np.zeros((2,1))
    for i in range(len(X_list)):
        x_i = np.array([[1,X_list[i]]])
        x_i = np.transpose(x_i)
        result[0] += np.array(  [ (Y_list[i]-  hypo_xi(theta, x_i ))*1   ]  ) # expected real value in square braces
        result[1] += np.array(  [ (Y_list[i]-  hypo_xi(theta, x_i ))*X_list[i]  ]  ) # expected real value in square braces
        
    return result/len(X_list)
    
    
    


# In[10]:


grad_J_theta(X_list,Y_list,theta)


# In[11]:


def cost(X_list,Y_list,theta):
    result = 0
    for i in range(len(X_list)):
        x_i = np.array([[1,X_list[i]]])
        x_i = np.transpose(x_i)
        result+= (Y_list[i]-  hypo_xi(theta, x_i ))**2
    return result/(2*len(X_list))

# def cost(x_vec,y_vec,theta):
#     abs_error =  y_vec - (x_vec @ theta)
#     #cost = (1.0/(2*len(x_vec)))* ( abs_error.T @ abs_error  )
#     cost = (1.0/2)* ( abs_error.T @ abs_error  )
#     return cost[0][0]


# In[12]:


#cost(X_list,Y_list,theta)


# In[13]:



# for i in range(10000):
#     theta += 0.01*grad_J_theta(X_list,Y_list,theta)
    
#     #print(theta)
#     print(cost(X_list,Y_list,theta))
  

plot_error_interval=[]    
plot_error_interval.append((theta[0][0],theta[1][0],  cost(X_list,Y_list,theta)) )    
learning_rate =0.01
prev_cost = cost(X_list,Y_list,theta)
theta += learning_rate*grad_J_theta(X_list,Y_list,theta)
curr_cost = cost(X_list,Y_list,theta)

print(curr_cost,prev_cost,curr_cost - prev_cost)
count =0
plot_theta_0=[]
plot_theta_1=[]
plot_cost=[]
start = time.time()
while  True:
    if abs(prev_cost - curr_cost) < 1e-16:
        break
    
    theta += learning_rate*grad_J_theta(X_list,Y_list,theta)
    count +=1
    #print(theta)
    prev_cost = curr_cost
    curr_cost = cost(X_list,Y_list,theta)
    #print(cost(X_list,Y_list,theta),prev_cost - curr_cost)
    end = time.time()
    if(end - start >= 0.2):
        plot_error_interval.append((theta[0][0],theta[1][0],  cost(X_list,Y_list,theta)) )
        start = time.time()
    
    
    print(theta[0],theta[1],cost(X_list,Y_list,theta))
    plot_theta_0.append(theta[0][0])
    plot_theta_1.append(theta[1][0])
    plot_cost.append(   cost(X_list,Y_list,theta)   )

#print(plot_error_interval[:5])    
# print(plot_theta_0)
# print(plot_theta_1)
# print(plot_cost)

print(count) 
    
# prev_theta = theta
# theta += 0.01*grad_J_theta(X_list,Y_list,theta)
# print(prev_theta,theta)
# count = 0
# while not(np.allclose(theta,prev_theta,atol=1e-6)):
#     theta += 0.01*grad_J_theta(X_list,Y_list,theta)
#     count +=1
#     print(theta)
#     prev_theta = theta
#     print(cost(X_list,Y_list,theta))
# print(count)    


# In[14]:


print(theta)


# In[31]:


# linearX = open("linearX.csv","r")
# X_list=[]
# for line in linearX:
#     #print(float(line.split("\n")[0]))    
#     X_list.append(float(line.split("\n")[0]))

#print(X_list)
x_0 = np.array([[1,X_list[0]]])
x_0 = np.transpose(x_0)
x_1 = np.array([[1,X_list[1]]])
x_1 = np.transpose(x_1)
x1 , y1 = X_list[0], hypo_xi(theta,x_0)
x2 , y2 = X_list[1], hypo_xi(theta,x_1)

hypo_xi(theta,x_i)



x1, y1 = [X_list[0], X_list[1]], [hypo_xi(theta,x_0), hypo_xi(theta,x_1)]
#plt.plot(x1,y1)

prediction =[]
for x in X_list:
    x_i = np.array([[1,x]])
    x_i = np.transpose(x_i)
    prediction.append(hypo_xi(theta,x_i))
#print(cost(X_list,Y_list,theta))
#print(Y_list)
#print(prediction)
                      
                      
#axes = plt.gca()
print(theta)
#axes.set_ylim([0.990,1.003])                     
plt.scatter(X_list,Y_list,color="blue")


x_axis = np.linspace(-2,5,10)
y_axis = theta[0][0] + theta[1][0]*x_axis

plt.ylim(0.985,1.005)
plt.xlabel("Normalized X")
plt.ylabel("Output")
plt.plot(x_axis,y_axis,color = 'red',label="hypo")
plt.legend()


# In[16]:


print(theta)


# In[ ]:





# In[17]:


# The function J
def J(a0, a1, x, y, m):
    J = 0
    for i in range(m):
        J += ((a0 + a1*x[i]) - y[i] )**2
    return J/(2*m)


# In[32]:


from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
a0 = np.linspace(0,2,100)
a1 = np.linspace(-1,1,100)
aa0, aa1 = np.meshgrid(a0, a1)
cost = J(aa0,aa1,X_list,Y_list,m=len(X_list))
ax.plot_surface(aa0, aa1, J(aa0,aa1,X_list,Y_list,m=len(X_list)),cmap = 'autumn',alpha = 0.6 )
ax.set_xlabel('Theta_0')
ax.set_ylabel('Theta_1')
ax.set_zlabel('Cost')



print(aa0[:10].shape)
print(aa1[:10].shape)
print(cost[:10].shape)

"""
err_t0 = []
err_t1 = []
err_val =[]
for t0,t1,val in plot_error_interval:
    err_t0.append(t0)
    err_t1.append(t1)
    err_val.append(val)
    

ax.plot( err_t0,err_t1,err_val ,'go' )
"""

import matplotlib.animation as animation

	

def animate( i ):
    line, = ax.plot( plot_theta_0[:i] , plot_theta_1[:i], plot_cost[:i], color="blue" )
    return line

ani = animation.FuncAnimation(fig, animate , interval = 200)

plt.show()
#ax.view_init(45,35)


# In[34]:


#a0 = np.linspace(-2,4, 50)
#a1 = np.linspace(-2,2.2, 50)
#J(aa0,aa1,x,y,m=len(x))
fig,ax = plt.subplots()
aa0, aa1 = np.meshgrid(a0, a1)
plt.xlabel("theta_0")
plt.ylabel("theta_1")

#plt.contour(aa0,aa1,J(aa0,aa1,X_list,Y_list,m=len(X_list)) )
plt.contour(aa0,aa1,cost,m=len(X_list)) 

#plt.plot( err_t0,err_t1,'go',linestyle='-'  )
plt.title( "Contour plot : rate = 0.001" )



# In[35]:


import matplotlib.animation as animation



def animate( i ):
    line, = ax.plot( plot_theta_0[:i] , plot_theta_1[:i], color="blue" )
    return line

ani = animation.FuncAnimation(fig, animate , interval = 200)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




