import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import math


l0 = 656.3

z = np.random.uniform(0.0,1.0)

def lobs(z,l0):
    lob = z*l0 + l0
    return lob

print("Pour z = ", z, "la longueur d'onde observee sera", lobs(z,l0),"nm")

def lon(ini,fin,num):
    lond = np.linspace(ini, fin, num, retstep = True)
    l = lond[0]
    step = lond[1]
    return l,step

l,step = lon(600,1400,1000)

#print("le tableau de longueur d'onde est:",l, "avec un pas qui separe deux longueurs: ",step)

dev = lobs(z,l0)*0.01
obs = lobs(z,l0)

def pdf(longueur, centre, deviation):
    y = stats.norm.pdf(longueur,centre,deviation)
    return y


""" DONNEES """

ysig = 30*pdf(l,obs,dev)

#plt.plot(l,ysig)
#plt.show()

bruit = np.random.normal(0,1,1000)
#plt.plot(l,bruit)
#plt.show()

data = ysig + bruit

#plt.plot(l,data)
#plt.show()


""" MODELS: Du signal et du bruit """

lmod,step = lon(0,60*step,60)
ymodel = pdf(lmod,(step*60/2),8)

#plt.plot(lmod,ymodel)
#plt.show()

bruitmod = np.random.normal(0,1,1000)

""" TRAITEMENT DU SIGNAL """

def cross(ymodel,ysig):
    sizemod = np.size(ymodel)
    sizesig = np.size(ysig)
    
    prod = []

    for i in range(sizesig-sizemod):
        p = np.dot(ymodel,ysig[i:(i+sizemod)])
        prod.append(p)
    return prod


prodat = cross(ymodel,data)
#plt.plot(prodat)
#plt.show()

prodbruit = cross(ymodel,bruitmod)
#plt.plot(prodbruit)
#plt.show()

devbr = np.std(prodbruit)
SNR = prodat/devbr

#plt.plot(SNR)
#plt.show()

index = np.argmax(SNR[SNR>3]) 
index+= 30
print(index)

l_found = l[index]
print("la longeur d'onde correspondante:" ,l_found)

def redshift(l_found, l0):
    redshift = abs((l_found - l0) /l0)
    return redshift

print(redshift(l_found, l0))













