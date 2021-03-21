# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:22:26 2020

@author: hp
"""

#coding:utf-8
import math as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#La fonction Rosenbrock
def Rosenbrock(X):
 return(X[0]**2 + 2*(X[1]**2)-2*X[0]*X[1]-4*X[0])

#Gradient a pas fixe
def gradient_f(X):
 return (-2*X[1]+2*X[0],-2*X[0]+4*X[1]-2)

def norme(u):
 x=u[0]
 y=u[1]
 return mp.sqrt(x**2+y**2)

# l enonce propose la fonction PasFixe avec seulement pour argument x (l initialisation) et beta le pas
#Par consequent kmax et epsilon (l erreur acceptee) seront donnes dans le programme

def PasFixe(x,beta):
 k=0
 x= [x]
 kmax=10000 #tres eleve juste pour ne pas bloquer la resolution mais eviter une boucle infinie ou tres longue
 epsilon=10**(-3)
 while norme(gradient_f(x[k]))>=epsilon and k<=kmax:
  a=x[k][0]+beta*(-gradient_f(x[k])[0])
  b=x[k][1]+beta*(-gradient_f(x[k])[1])
  A=(a,b)
  x.append(A)
  k+=1
  
 return (k,x[-1],norme(gradient_f(x[-1])),x)

PF = PasFixe([0,1],0.01)
print (" PasFixe : k = ",PF[0]," et [x,y] = ",PF[1], " et Epsilon = ",PF[2])

def gradient_f(X):
 return ([2*X[0]-2+40*X[0]**3-40*X[0]*X[1],-20*X[0]**2+20*X[1]])

def norme(u):
 x=u[0]
 y=u[1]
 return mp.sqrt(x**2+y**2)

# Gradient a pas optimal

def Omega (X,alpha):
 return ([X[0]-alpha*gradient_f(X)[0],X[1]-alpha*gradient_f(X)[1]])

def SectionDoree(X0,a,b,tolerance): #Recherche du pas optimal
 tau = (1+np.sqrt(5))/2
 it = 0
 err = b-a
 while np.abs(err)>tolerance :
  aprime = a + (b-a)/(tau*tau)
  bprime = a + (b-a)/tau
  c , d = Omega(X0,aprime) , Omega(X0,bprime)
  if c > d :
    a = aprime
  elif c < d :
       b = bprime
  else :
       a , b = aprime , bprime

  err = b-a   
  it = it + 1
 return (a + b)/2

alpha = SectionDoree([0,1],0,1,0.01)

def PasOptimal (X0,alpha):

 x=X0[0]
 y=X0[1]
 listeXY = [[x,y]]
 k=0
 kmax=10000

 gradx,grady = gradient_f([x,y])

 while (norme(gradient_f([x,y]))>0.01) and (k<=kmax):
   X = [x,y]
   x,y = Omega(X,alpha)
   k+=1
   gradx,grady = gradient_f([x,y])
   listeXY.append([x,y])
   X = [x,y]

   a=X0[0]-alpha*gradient_f(X0)[0]
   b=X0[1]-alpha*gradient_f(X0)[1]

 Resp=Rosenbrock([a,b])

 return (X,k,norme(gradient_f([x,y])),listeXY,Resp)

PO = PasOptimal([0,1],alpha)
print (" PasOptimal : [x,y] = " ,PO[0]," et k = ",PO[1], " et Epsilon = ",PO[2])

# Affichage des courbes des iteres pour les 2 methodes 
def CreerlistepointsX(liste):
 A = []
 for i in range (len(liste)):
    A.append(liste[i][0])

 return (A)

def CreerlistepointsY(liste):
 B= []
 for i in range (len(liste)):
   B.append(liste[i][1])
   
 return (B)

# Création de la liste contenant les valeurs de X et Y
A = CreerlistepointsX(PasFixe([0,1],0.01)[3])
B = CreerlistepointsY(PasFixe([0,1],0.01)[3]) 
C = CreerlistepointsX(PasOptimal([0,1],alpha)[3]) 
D = CreerlistepointsY(PasOptimal([0,1],alpha)[3])

plt.plot(A,B,"o",color="b", label = " Gradient à pas fixe ")
plt.plot(C,D,"o",color="g", label = " Gradient à pas optimal ")
plt.legend()
plt.title(" Convergence des solutions ")
plt.xlabel(" X ")
plt.ylabel(" Y ")

ax = Axes3D(plt.figure())
X = np.linspace(-10,10,50)
Y = np.linspace(-10,10,50)
X, Y = np.meshgrid(X, Y)
Z = Rosenbrock([X,Y])
ax.plot_surface(X, Y, Z)
plt.title(" Fonction de quadratique ")
plt.xlabel(" X ")
plt.ylabel(" Y ")
#ax.plot(A,B,'o',color='r', label = " Gradient à pas fixe ")
#ax.plot(C,D,'o',color='y', label = " Gradient à pas optimal ")
plt.legend()
plt.show() # affichage de la nappe
