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
    sinu = amplitude*np.sin(pulsation*tps[tps<1])
    return sinu
    
signal = sinuso(tps, amplitude, pulsation)
    
plt.plot(signal)
plt.show()



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


#plt.plot(frequence, dens)
#plt.show()

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
plt.plot(a)
plt.show()

""" Génération de la matrice de covariance du bruit """

Cn = []
for i in range(1000):
    bruit_color = gener_color(dens2)
    Cn += np.dot((np.transpose(bruit_color)), bruit_color)
    print(np.size(bruit_color))
Cn /= np.size(tps)

print(Cn)















