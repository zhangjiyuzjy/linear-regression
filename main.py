import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import linear_reg as lr


#Get the data from the txt file
data= sp.genfromtxt('ex1data1.txt',delimiter=",")

#get the input data set and output vectors
xd=data[:,0]
y=data[:,1]
m=y.size

#define regression parameters
iterations=1500
alpha =0.01
#Plot the data to get a visual understanding
print "Now I am plotting data"
plt.figure(1)
plt.scatter(xd,y) 
plt.title("population vs profit")
plt.ylabel("profit in $10000")
plt.xlabel("population in 10000") 
plt.grid()

Y=np.reshape(y,(m,1))
x=[np.ones(m),xd]
#initialize the theta vector to zero
theta_ini= np.zeros((2,1))

#Compute the cost function with initial Theta values
cost=lr.computecost(x,Y,theta_ini)
print cost

#Compute the Theta which minimizes the cost function
theta,J_his=lr.gradient(x,Y,np.zeros((2,1)),alpha,iterations)
hypothesis=np.dot(np.transpose(x),theta)
plt.plot(xd,hypothesis,'-')

print "this is  optimized Theta : \n"
print theta

# plot the cost versus iteration curve to see if the gradient Descent i s working properly
plt.figure(2)
plt.plot(np.arange(0,1500,1),J_his)
plt.show()


