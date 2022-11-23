import numpy as np
import matplotlib.pyplot as plt


tps = np.linspace(0, 10, 1000, retstep = True)
temps = tps[0]
pas = tps[1]


""" Signal Sinusoidal """
 
def sinusoidal(temps, amplitude_init, pulsation_init):
    """
    Génére un signal dont l'amplitude et la frequence augmentent en
    fonction du temps de manière linéaire

    Args:
        temps (array): temps
        amplitude_init (float): Amplitude initiale du signal
        pulsation_init (float): Fréquence initiale du signal
    Returs:
        array : signal sinusoïdal
    """
    amplitude = amplitude_init*temps
    pulsation = pulsation_init*temps

    signal = amplitude*np.sin(pulsation*temps)
    return signal

""" Signal mesuré """
temps1 = temps[0:100]

amp_init = 2
puls_init = 30 # en rad/s

signal = sinusoidal(temps1, amp_init, puls_init)

plt.plot(signal)
plt.title('signal sinusoïdale mesuré')
plt.show()


print("la fréquence d'échantillonnage est : ", pas, "s.", "C'est à dire" , 1/pas, "Hz")

frequence1 = np.fft.fftfreq(np.size(temps1), pas)

plt.plot(frequence1)
plt.title("frequence d'échantillonage dans l'espace de Fourier")
plt.show()

# Transformé de Fourier de notre signal mesuré:
signal_fourier = np.fft.fft(signal)
plt.plot(frequence1, np.real(signal_fourier))
plt.title("signal dans l'espace de Fourier")
plt.show()

""" Densité spectrale de puissance d'un bruit coloré """

def psd(frequence, gamma):
    """
    Calcule la densité spectrale de puissance d'un bruit coloré

    Args:
        frequence (array): frange frequences positives à partir desquelles générer un bruit coloré
        gamma (float): indice loi de puissance

    Returns:
        array : loi de puissance
    """
    p = ((frequence[frequence>0])/10)**(gamma) + 1
    return p

gamma = -1.55
dens1 = psd(frequence1, gamma)
plt.plot(dens1)
plt.title("PSD avec $\gamma$ = -1.55 pour 1s")
plt.show()

def miroir(densite):
    """
    Génére la partie miroir d'une fonction et la rajoute à cette fonction.
    Ajoute aussi le 0.

    Args:
        densite (array): fontion à laquelle on veut ajouter sa partie miroir.

    Returns:
        array : fonction avec partie miroir et 0
    """
    densite = np.insert(densite, 0,  0)
    densite = np.concatenate((densite, np.flip(densite)))
    return densite

dens1 = psd(frequence1, gamma)
dens1 = miroir(dens1)
plt.plot(frequence1, dens1)
plt.title("PSD avec PSD miroir")
plt.show()

# Maintenant le bruit de toute la mesure (10s):

frequence = np.fft.fftfreq(np.size(temps), pas)

dens = psd(frequence, gamma)
plt.plot(dens)
plt.title("PSD du bruit coloré pour 10 s")
plt.show()

dens = miroir(dens)
plt.plot(frequence, dens)
plt.title("PSD avec PSD miroir")
plt.show()


""" Espace de Fourier pour trouver le bruit coloré """

def bruit(densite):
    """
    Crée du bruit coloré à partir de sa densité spectrale de puissance.

    Args:
        densite (array): densité spectrale de puissance du bruit coloré dans l'espace de Fourier.

    Returns:
        array : le bruit coloré dans l'espace réel
    """
    bruit_blanc = np.random.normal(0,1,1000)
    fourier_blanc = np.fft.fft(bruit_blanc)

    fourier_color = fourier_blanc*np.sqrt(densite)
    bruit_color = np.real(np.fft.ifft(fourier_color))

    return bruit_color

""" Introduction aleatoire de la signal dans le bruit coloré """
pos = np.random.randint(1000-100)

def signalsurbruit(densite, signal, position):
    """
    Introduit un signal dans du bruit coloré

    Args:
        densite (array): densité spectrale de puissance du bruit coloré
        signal (array): signal
        position (int): position où mettre la signal dans le bruit

    Returns:
        array: data : signal + bruit
    """
    bruit_color = bruit(densite)
    bruit_color[position:position+np.size(signal)] += signal
    
    return bruit_color

signal_bruit = signalsurbruit(dens, signal, pos)
plt.plot(signal_bruit)
plt.title("bruit coloré avec signal introduit de manière aléatoire")
plt.show()

# On essaie une amplitude plus grande pour vérifier que le signal est dedans:

big_amp_init = 30

big_signal = sinusoidal(temps1, big_amp_init, puls_init)

big_signal_bruit = signalsurbruit(dens,big_signal,pos)

plt.plot(big_signal_bruit)
plt.title("bruit coloré avec grand signal introduit de manière aléatoire")
plt.show()

""" Modélisation signal pour le filtre """

amp_init_model = 1
signal_model = sinusoidal(temps1, amp_init_model, puls_init)

plt.plot(signal_model)
plt.title("signal modélisé avec amplitude plus petite")
plt.show()

""" Modélisation bruit pour le filtre """

def covariance(densite, temps):
    """
    Calcule l'inverse de la matrice de covariance de 1000 réalisations du bruit en 1 s

    Args:
        densite (array): densité spectrale de puissance du bruit coloré
        temps (array): plage de temps

    Returns:
        array : Matrice de covariance (Cn) et son inverse (cov)
    """
    Cn = np.zeros((100,100))
    for i in range(1000):
        bruit_color = bruit(densite)
        Cn += np.dot(np.transpose(bruit_color[None, 0:100]), bruit_color[None, 0:100])
    Cn /= np.size(temps)
    cov = np.linalg.inv(Cn)
    return cov, Cn

cov, Cn = covariance(dens, temps)

# Vérification des propietés de la matrice covariance du bruit :

if (Cn.all() == np.transpose(Cn).all()):
    print("La matrice de covariance est égale à la matrice de covariance transposée")

I = np.identity(100)
prod = np.dot(Cn,cov)
prod = prod.astype(int)

if prod.all() == I.all():
    print("La matrice de covariance fois son inverse donne l'identité")


""" FILTRE """
filtre = np.dot(cov, signal_model)


""" Application filtre """

def cross(filtr,data):
    """
    Fait le produit scalaire morceau par morceau du temps
    entre le filtre et le signal mesuré (signal + bruit)

    Args:
        filtr (array): filtre
        data (array): signal + bruit

    Returns:
        array : signal filtré
    """
    sizemod = np.size(filtr)
    sizesig = np.size(data)

    prod = []
    for i in range(sizesig-sizemod):
        p = np.dot(filtr,data[i:(i+sizemod)])
        prod.append(p)
    return prod

""" Corrélations croisés entre modèle et données (signal + bruit)"""

prodat = cross(filtre, signal_bruit)
plt.plot(prodat)
plt.title("signal + bruit après filtrage")
plt.show()

""" 1000 réalisations de bruit filtrés """

def bruitfiltre(densite,filtre):
    """
    Filtre 1000 réalisations de bruit

    Args:
        densite (array): densité spectrale de puissance
        filtre (array): filtre

    Returns:
        array : (bruit_filtre) bruit filtré en moyenne
        float : (devbr) écart type bruit filtré (1000 réalisations)
    """
    bruit_filtre = np.zeros(900)
    devbr = 0
    for i in range(1000):
        bruit_color = bruit(densite)
        prodbruit = cross(filtre, bruit_color)
        bruit_filtre += prodbruit
        dev = np.std(prodbruit)
        devbr+= dev
    bruit_filtre = bruit_filtre/1000
    devbr = devbr/1000
    return bruit_filtre, devbr

def bruitfiltreplot(densite,filtre):
    """
    Plot les 1000 réalisations de bruit filtré

    Args:
        densite (array): densité spectrale de puissance
        filtre (array): filtre

    Returns:
        array : bruit filtré en moyenne
    """
    plt.figure()
    bruit_filtre = np.zeros(900)
    for i in range(1000):
        bruit_color = bruit(densite)
        prodbruit = cross(filtre, bruit_color)
        plt.title("1000 réalisations de bruit filtré")
        plt.plot(prodbruit)
    plt.show()
    return bruit_filtre


bruit_filtre, devbr = bruitfiltre(dens,filtre)
plot_bruit_filtre = bruitfiltreplot(dens,filtre)

""" Rapport signal sur bruit (SNR)"""

SNR = prodat/devbr
plt.plot(SNR)
plt.title("Rapport signal sur bruit (SNR)")
plt.show()

""" Détection si SNR > 3 """

def position_SNR_max(SNR):
    """
    Cherche la position de la signal avec une sensibilité de SNR > 3

    Args:
        SNR (array): Rapport signal sur bruit
    
    Returns:
        int : position du maximum du SNR s'il est > 3
    """
    if np.max(SNR)<3:
        print("Pas de signal détecté")
        return None
    
    index = np.argmax(SNR)
    
    return index

"""On cherche la position et on la compare avec la position aléatoire où on l'avait mis """
position = position_SNR_max(SNR)
print("la position trouvé avec le filtre est : ", position)
print("la position aléatoire où on avait mis la signal dans le bruit est : ", pos)

# Pour voir la comparaison des gamma aller voir le Jupyter Notebook associé au signal complexe.

# Des fois la position diffère d'un point, c'est parce que notre filtre est un peu plus petit que notre signal, mais la plupart des fois c'est juste




