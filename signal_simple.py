import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import math


"""
Un objet lointain émet une raie de longueur 656,3 nm
Nous observons cette raie décalée : quel est son redshift?
Pour cela nous devons déterminer la position de la raie observée sur le signal reçu
"""




### Fonctions annexes :

# calcule la longueur d'onde observée
def lobs(z,l0):
    lob = z*l0 + l0
    return lob
    
# calcule le redshift à partir de la longueur d'onde observée
def redshift(obs, l0):
    redshift = abs(obs - l0)/l0
    return redshift
    
# créer n valeurs sur une plage de ini à fin
def lon(ini,fin,n):
    lond = np.linspace(ini, fin, n, retstep = True)
    l = lond[0]
    step = lond[1]
    return l,step

# calcule la fonction de densité de probabilité sur une plage longueur
# d'un signal centré sur centre et de déviation dev 
def pdf(longueur, centre, dev):
    y = stats.norm.pdf(longueur,centre, dev)
    return y
    
# corrélations croisées entre des vraies données et un signal modèle
def cross(ymodel,ysig):
        sizemod = np.size(ymodel)
        sizesig = np.size(ysig)
        
        prod = []
    
        for i in range(sizesig-sizemod):
            p = np.dot(ymodel,ysig[i:(i+sizemod)])
            prod.append(p)
        return prod
    
    
    
    
### Fonction principale calculant le signal sur bruit
# En argument: l'amplitude du signal, l'écart-type du signal, l'amplitude du bruit

def traitement_signal_simple(obs, amp_signal, dev_signal, amp_bruit):
    
    """ Création du signal """
    
    l,step = lon(600,1400,1000)
    
    
    dev = dev_signal*0.01*obs
    
    """ DONNEES """
    
    ysig = amp_signal*pdf(l,obs,dev)
    
    plt.plot(l,ysig)
    plt.title("Signal seul")
    plt.xlabel("Longueur d'onde en nm")
    plt.ylabel("Pdf du signal")
    plt.show()
    
    
    bruit = amp_bruit*np.random.normal(0,1,1000)
    plt.plot(l,bruit)
    plt.title("Bruit blanc gaussien")
    plt.xlabel("Longueur d'onde en nm")
    plt.ylabel("Déviation standard")
    plt.show()
    
    data = ysig + bruit
    
    plt.plot(l,data)
    plt.title("Signal + bruit")
    plt.xlabel("Longueur d'onde en nm")
    plt.ylabel("Déviation standard")
    plt.show()
    
    
    """ MODELS: Du signal et du bruit """
    
    lmod,step = lon(0,60*step,60)
    ymodel = pdf(lmod,(step*60/2),8)
    
    bruitmod = amp_bruit*np.random.normal(0,1,1000)
    
    """ TRAITEMENT DU SIGNAL """
    
    # Corrélations croisées modèle/données
    prodat = cross(ymodel,data)
    
    # Corrélations croisées modèle/bruit
    prodbruit = cross(ymodel,bruitmod)
    
    
    devbr = np.std(prodbruit)
    SNR = prodat/devbr
    
    plt.plot(SNR)
    plt.title("Signal sur bruit")
    plt.show()
    return SNR
 
# Fonction traitement le signal-sur-bruit:
# détecte un signal seulement au-dessus de 3 sigma
def traitement_SNR(obs, SNR):
    
    SNR = SNR[SNR>3]
    
    if not np.any(SNR):
        print("Pas de signal détecté")
        return 0
        
    index = np.argmax(SNR[SNR>3]) 
    index+= 30
    print(index)
    
    l,step = lon(600,1400,1000)
    l_found = l[index]
    print("la longueur d'onde trouvée:", l_found)
    z = redshift(obs, l_found)
    print("Le redshift trouvé:" ,z)
    
    return 0
    
    
    
# Données initiales:
    
l0 = 656.3
z = np.random.uniform(0.0,1.0)
obs = lobs(z,l0)
print("La longueur d'onde émise est :", l0, "nm")
print("Pour z = ", z, "la longueur d'onde observee sera", obs,"nm")

print("On cherche à retrouver le redshift à partir d'un signal de la longueur d'onde observée")

# Traitement d'un signal simple pour 4 cas:

print("\nCas initial:")
SNR = traitement_signal_simple(obs, 20, 1, 1)
traitement_SNR(obs, SNR)

print("\nCas où on augmente l'amplitude du signal:")
SNR = traitement_signal_simple(obs, 100, 1, 1)
traitement_SNR(obs, SNR)

print("\nCas où on diminue l'écart-type du signal:")
SNR = traitement_signal_simple(obs, 20, 0.1, 1)
traitement_SNR(obs, SNR)

print("\nCas où on augmente le bruit:")
SNR = traitement_signal_simple(obs, 20, 1, 2)
traitement_SNR(obs, SNR)

print("\nEn conclusion:\
        \nLa détection est meilleure pour un bruit faible, une amplitude \
du signal haute, et une déviation standard du signal faible")


