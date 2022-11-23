import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
from scipy.signal import correlate
import time
#from codecarbon import EmissionsTracker



#tracker = EmissionsTracker()
#tracker.start()



""" Fonctions """

def cross(ymodel,ysig):
        sizemod = np.size(ymodel)
        sizesig = np.size(ysig)
        
        prod = []
    
        for i in range(sizesig-sizemod):
            p = np.dot(ymodel,ysig[i:(i+sizemod)])
            prod.append(p)
        return prod

def sinusoidal(temps, amplitude_init, pulsation_init):
    
    amplitude = amplitude_init*temps
    pulsation = pulsation_init*temps
    
    signal = amplitude*np.sin(pulsation*temps)
    return signal

def psd(frequence, gamma):
    p = ((frequence[frequence>0])/10)**(gamma) + 1
    return p

def miroir(densite):
    densite = np.insert(densite, 0,  0)
    densite = np.concatenate((densite, np.flip(densite)))
    return densite

# Fonction ressortant un bruit color (1000 valeurs)
def bruit(densite):
    bruit_blanc = np.random.normal(0,1,1000)
    fourier_blanc = np.fft.fft(bruit_blanc)
    
    fourier_color = fourier_blanc*np.sqrt(densite)
    bruit_color = np.real(np.fft.ifft(fourier_color))
    return bruit_color


# Créer un bruit coloré, et ajoute le signal à la position indiquée
def signalsurbruit(densite, signal, position):
    bruit_color = bruit(densite)
    bruit_color[position:position+np.size(signal)] += signal
    return bruit_color

# Fonction retournant la matrice de covariance (Cn) et la matrice inverse cov
def covariance(densite, temps):
    Cn = np.zeros((100,100))
    n = 1000
    for i in range(n):
        bruit_color = bruit(densite)
        Cn += np.dot(np.transpose(bruit_color[None, 0:100]), bruit_color[None, 0:100])
    Cn /= n
    cov = np.linalg.inv(Cn)
    return cov, Cn

def cross(ymodel,ysig):
        sizemod = np.size(ymodel)
        sizesig = np.size(ysig)
        
        prod = []
    
        for i in range(sizesig-sizemod):
            p = np.dot(ymodel,ysig[i:(i+sizemod)])
            prod.append(p)
        return prod
    
def bruitfiltre(densite,filtre):
    bruit_filtre = np.zeros(900)
    devbr = 0
    n = 1000
    for i in range(n):
        bruit_color = bruit(densite)
        prodbruit = cross(filtre, bruit_color)
        bruit_filtre += prodbruit
    bruit_filtre = bruit_filtre/n
    devbr =  np.std(bruit_filtre)
    return bruit_filtre, devbr

    
def bruitfiltre_correlate(densite,filtre):
    bruit_filtre = np.zeros(901)
    devbr = 0
    n = 1000
    for i in range(n):
        bruit_color = bruit(densite)
        prodbruit = correlate(bruit_color, filtre, mode='valid')
        bruit_filtre += prodbruit
    bruit_filtre = bruit_filtre/n
    devbr = np.std(bruit_filtre)
    return bruit_filtre, devbr



""" Premier cas : le signal simple """


# Valeurs

l,step = np.linspace(600,1400,1000, retstep = True)
    
l0 = 656.3
z = np.random.uniform(0.0,1.0)
obs = z*l0 + l0
    
dev = 0.01*obs
    
# DONNEES

ysig = 20*stats.norm.pdf(l,obs,dev)
bruit_simple = np.random.normal(0,1,1000)
data = ysig + bruit_simple

# MODELS: Du signal et du bruit

lmod,step = np.linspace(0,60*step,60, retstep = True)
ymodel = stats.norm.pdf(lmod,(step*60/2),8)

bruitmod = np.random.normal(0,1,1000)
    
### Traitement du signal simple par notre code initial :


start = time.time()  
    
# Corrélations croisées modèle/données

prodat = np.zeros(940)
for i in range(1000):
    prodat += cross(ymodel,data)
prodat /= 1000

# Corrélations croisées modèle/bruit

prodbruit = cross(ymodel,bruitmod)
    
devbr = np.std(prodbruit)
SNR = prodat/devbr

end = time.time()
t = end - start
print("Temps mis par notre code pour le signal simple:", t)



### Traitement du signal simple par la fonction correlate :

start_2 = time.time()

prodat_2 = correlate(data, ymodel, mode='valid')
# Le signal est tronqué, il faut utilisé le mode valid pour ne pas avoir un décalage
    
# Corrélations croisées modèle/bruit
prodbruit_2 = correlate(bruitmod, ymodel, mode='valid')
    
devbr_2 = np.std(prodbruit_2)
SNR_2 = prodat_2/devbr_2

end_2 = time.time()
t_2 = end_2 - start_2
print(" Temps mis par la fonction correlate pour le signal simple:", t_2)




### Comparaison des deux méthodes:

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Signal sur bruit - Cas simple')

ax1.plot(SNR_2, 'y')
ax1.set_ylabel('Fonction correlate')

ax2.plot(SNR, 'b')
ax2.set_ylabel('Notre code')

plt.show()








""" Deuxième cas : le signal complexe """

print("Cas complexe")



### Données initiales :

# Plage temporelle de 1000 valeurs, de 0 à 10s
tps = np.linspace(0, 10, 1000, retstep = True)
temps = tps[0]
pas = tps[1]
print("Le pas:", pas)

# Plage temporelle de 100 valeurs, de 0 à 1s
temps1 = temps[0:100]

# Position aléatoire entre 0 et 900
pos = np.random.randint(1000-100)




### Création du signal et bruit :

# Signal (de 1s)
amp_init = 1
puls_init = 30 # en rad/s

signal = sinusoidal(temps1, amp_init, puls_init)

# Bruit coloré (de 10s)
gamma = -1.55
frequence = np.fft.fftfreq(np.size(temps), pas)
bruit_color = psd(frequence, gamma)
bruit_color = miroir(bruit_color)


# Bruit + signal (data, de 10s, 1000 valeurs)
signal_bruit = signalsurbruit(bruit_color, signal, pos)


# Signal "modèle" de 100 valeurs

amp_init_model = 1
signal_model = sinusoidal(temps1, amp_init_model, puls_init)
# Inverse covariance
cov, Cn = covariance(bruit_color, temps)

# Filtre (100 valeurs)
filtre = np.dot(cov, signal_model)


### Traitement du signal avec notre code:

# Signal
start = time.time() 

prodat = cross(filtre, signal_bruit)

#Bruit
bruit_filtre, devbr = bruitfiltre(bruit_color,filtre)

SNR_3 = prodat/devbr

end = time.time()
t = end - start
print("\nTemps mis par notre code pour le signal complexe:",t)



### Traitement du signal avec la fonction correlate:

start_2 = time.time()

# Corrélations croisés modèle/signal

prodat_2 = correlate(signal_bruit, filtre, mode='valid')
    
# Corrélations croisées modèle/bruit

prod_bruit2, devbr_2 = bruitfiltre_correlate(bruit_color,filtre)


# SNR 

SNR_4 = prodat_2/devbr_2


end_2 = time.time()
t_2 = end_2 - start_2

print("Temps mis par la fonction correlate pour le signal complexe:", t_2)




### Comparaison des deux méthodes:

print("\nPour notre code, le signal est à la position temporelle:",np.argmax(np.abs(SNR_3)), "\
        \nPour correlate :", np.argmax(np.abs(SNR_4)))

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Signal sur bruit - Cas complexe')

ax1.plot(SNR_4, 'y')
ax1.set_ylabel('Fonction correlate')

ax2.plot(SNR_3, 'b')
ax2.set_ylabel('Notre code')

plt.show()




#tracker.stop()
