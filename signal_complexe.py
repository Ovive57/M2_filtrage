import numpy as np
import matplotlib.pyplot as plt


temps = np.linspace(0,10,1000, retstep = True)

tps = temps[0]
pas = temps[1]


""" Signal Sinusoidal """




temps2 = np.arange(0,1, pas)

amp_init = 2
amplitude = amp_init*temps2
    

puls_init = 30 # en rad/s
pulsation = puls_init*temps2

print("La fréquence d'échantillonage en Hz est:", pas)  #en Hz

def sinuso(tps, amplitude, pulsation):
    sinu = amplitude*np.sin(pulsation*tps) # J'ai changé ici pour le mettre plus général, ça pose des problèmes plus tard.
    return sinu
    
signal = sinuso(temps2, amplitude, pulsation)
    
#plt.plot(signal)
#plt.show()



""" Frequences de fourier du signal """


frequence = np.fft.fftfreq(np.size(temps2), pas)

print(frequence)

#plt.plot(frequence)
#plt.show()

signal_fourier = np.fft.fft(signal)

#plt.plot(frequence, np.real(signal_fourier))
#plt.show()


""" PSD : bruit coloré dans l'espace de Fourier """


gamma = -1.55
def psd(frequence, gamma):
    p = ((frequence[frequence>0])/10)**(gamma) + 1
    return p
    
dens = psd(frequence, gamma)
#plt.plot(dens)
#plt.show()

dens = np.insert(dens, 0,  0)
dens = np.concatenate((dens, np.flip(dens)))


plt.plot(frequence, dens)
plt.show()

"""" Modélisation d'un autre bruit :"""


frequence_1000 = np.fft.fftfreq(np.size(tps), pas)

dens2 = psd(frequence_1000, gamma)
dens2 = np.insert(dens2, 0,  0)
dens2 = np.concatenate((dens2, np.flip(dens2)))


# fonction générant un bruit coloré
def gener_color(dens2):
    bruit_blanc = np.random.normal(0,1,1000)

    fourier_blanc = np.fft.fft(bruit_blanc)

    fourier_color = fourier_blanc*np.sqrt(dens2)

    bruit_color = np.real(np.fft.ifft(fourier_color))
    
    return bruit_color

""" Créer signal + bruit """


pos = np.random.randint(1000-100)
#print(pos)
bc = gener_color(dens2)
bc[pos:pos+np.size(signal)]+= signal
a = bc
#print(a)
#plt.plot(a)
#plt.show()

""" Génération de la matrice de covariance du bruit """

Cn = np.zeros((100,100))
bruit_color = gener_color(dens2)

#print(np.shape(bruit_color))
#print(np.shape(np.dot(np.transpose(bruit_color[None, 0:100]), bruit_color[None, 0:100])))

#Cn += np.dot(bruit_color[None; 0:100], bruit_color[None, :])

# On fait la moyenne pour 1000 cad l'espérance
for i in range(1000):
    bruit_color = gener_color(dens2)
    Cn += np.dot(np.transpose(bruit_color[None, 0:100]), bruit_color[None, 0:100])
Cn /= np.size(tps)

#print(Cn)
Cov = np.linalg.inv(Cn)

#print(Cov)

# Test de inverse = transposée
if (Cn.all() == np.transpose(Cn).all()):
    print("ok")


# On crée un autre signal = signal modèle, pour le filtre:

amp_init_model = 3
amplitude_model = amp_init*temps2
    
# c'est le g du cours:
signal_model = sinuso(tps, amplitude_model, pulsation)

filtre = np.dot(Cov, signal_model)

#print(filtre)
#print(np.shape(filtre))


def cross(ymodel,ysig):
    sizemod = np.size(ymodel)
    sizesig = np.size(ysig)
    
    prod = []

    for i in range(sizesig-sizemod):
        p = np.dot(ymodel,ysig[i:(i+sizemod)])
        prod.append(p)
    return prod


prodat = cross(filtre,a)

prodbruit = cross(filtre,bruit_color)

devbr = np.std(prodbruit)
SNR = prodat/devbr

print(frequence_1000)
print(tps)
plt.plot(SNR)
plt.show()

index = np.argmax(SNR[SNR>4]) 
index+= 50
print(index)

print(frequence)
l_found = tps[index]
print("le signal sort au bout de :" ,l_found, " s")



