import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Funnktionen

def lade_json_als_array(pfad):
    with open(pfad, 'r') as f:
        return np.array(json.load(f))

def analysiere_daten(daten, name):
    print(f"Analyse für die Datei {name}:")
    print(f"Anzahl der Datenpunkte: {daten.shape[0]}")
    print(f"Anzahl der Merkmale: {daten.shape[1]}\n")
    print("Mittelwerte:", np.mean(daten, axis=0))
    print("Standardabweichungen:", np.std(daten, axis=0))
    print("Minimalwerte:", np.min(daten, axis=0))
    print("Maximalwerte:", np.max(daten, axis=0))
    print("-----------------------------------------------------")

def plot(daten, titel):
    plt.figure(figsize=(8, 6))
    plt.scatter(daten[:, 0], daten[:, 1], alpha=0.5, s=5)
    plt.title(titel)
    plt.xlabel("Merkmal 0")
    plt.ylabel("Merkmal 1")
    plt.grid(True)
    plt.show()

def min_max_normalisieren(daten):
    minimums = np.min(daten, axis=0)
    maximums = np.max(daten, axis=0)
    normalisierte_daten = (daten - minimums) / (maximums - minimums)
    return normalisierte_daten

# Daten laden lassen
x0 = lade_json_als_array("data/x/x0.json")
x1 = lade_json_als_array("data/x/x1.json")
x2 = lade_json_als_array("data/x/x2.json")

# Daten werden hier Analysiert
analysiere_daten(x0, "x0.json")
analysiere_daten(x1, "x1.json")
analysiere_daten(x2, "x2.json")

# Daten werden hier geplottet
plot(x0, "Scatter Plot für x0.json")
plot(x1, "Scatter Plot für x1.json")
plot(x2, "Scatter Plot für x2.json")

# Normalisierung
x0_norm = min_max_normalisieren(x0)
x1_norm = min_max_normalisieren(x1)
x2_norm = min_max_normalisieren(x2)

print("Normalisierung abgeschlossen.")

plot(x1_norm, "Scatter Plot für normalisiertes x1.json")

# PCA auf x1
pca = PCA(n_components=2)
x1_pca = pca.fit_transform(x1_norm)

print(f"PCA abgeschlossen: Neue Form {x1_pca.shape}")
plot(x1_pca, "PCA: x1.json auf 2D reduziert")

# Dichteanalyse x2
plt.figure(figsize=(8, 6))
plt.hist2d(x2[:, 0], x2[:, 1], bins=50, cmap='plasma')
plt.colorbar(label='Punktdichte')
plt.title("Dichte-Heatmap für x2.json")
plt.xlabel("Merkmal 0")
plt.ylabel("Merkmal 1")
plt.grid(True)
plt.show()

# Ausreißeranalyse x0
schwelle = 50

# Betrag der ersten beiden Merkmale berechnen
merkmal_0_betrag = np.abs(x0[:, 0])
merkmal_1_betrag = np.abs(x0[:, 1])

# Maske: nur die Punkte, bei denen beide Merkmale unter der Schwelle liegen
maske = (merkmal_0_betrag < schwelle) & (merkmal_1_betrag < schwelle)

# Aufteilen in Kern und Ausreißer
x0_kern = x0[maske]
x0_ausreisser = x0[~maske]

print(f"Kernpunkte: {len(x0_kern)}")
print(f"Ausreißer: {len(x0_ausreisser)}")

plt.figure(figsize=(8, 6))
plt.scatter(x0_kern[:, 0], x0_kern[:, 1], alpha=0.5, s=10, label='Kern')
plt.scatter(x0_ausreisser[:, 0], x0_ausreisser[:, 1], alpha=1.0, s=50, label='Ausreißer')
plt.title("x0.json: Kern vs. Ausreißer")
plt.xlabel("Merkmal 0")
plt.ylabel("Merkmal 1")
plt.legend()
plt.grid(True)
plt.show()


