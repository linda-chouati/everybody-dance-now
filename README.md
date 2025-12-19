# Everybody Dance Now - Transfert de Mouvement 

**Master 2 Intelligence Artificielle - Université Lyon 1** **UE :** Apprentissage Profond et Image (Encadré par Alexandre Meyer)  
**Étudiante :** Chouati Linda (p1805862)

## Description du Projet

Ce projet implémente plusieurs approches de **transfert de mouvement**. L'objectif est de générer une vidéo d'une personne cible reproduisant les mouvement d une personne source.

Le code consiste a extraire les squelettes d une vidio source et à entraîner un réseau de neurones pour synthétiser l'image de la personne cible correspondant à ces poses.

### Fonctionnalités principales
* Extraction de squelettes via **Mediapipe**.
* Génération d images via plusieurs architectures (NN simple, U-Net, GAN).
* Application interactive via **streamlit**.

---

## Architectures et Approches Implémentées

Plusieurs méthodes ont été exploré, allant du plus simple au plus complexe. 

### 1. Approche Nearest Neighbor => plus proche voisins 
* **Fichier :** `GenNearest.py`
* **Principe :** Recherche dans le dataset de l image dont le squelette est le plus proche du squelette cible.
* **Résultat : je trouve le résultat plutot satisfaisant pour une méthode tres simple, après on voit assez clairement aussi que y'a pas de généralisation, que c'est saccadé.

### 2. Réseaux de Neurones  (Vanilla NN)
* **Fichier :** `GenVanillaNN.py`
* **Approche 1 : :**
    * *Entrée :* vecteur de 26 coordonnée
    * *Architecture :* couches *Fully Connected* avec **SEBlocks (Squeeze-and-Excitation)** pour l'attention par canal et *Skip Connections*.
* **Approche 2 :**
    * *Entrée :* image RGB (64x64) représentant le squelette dessiné
    * *Architecture :* type **U-Net** (ecodeur-décodeur) enrichi avec :
        * **Self-Attention 2D** pour capturer les dépendances globales
        * **Residual Blocks** pour faciliter l'apprentissage profond

### 3. Generative Adversarial Networks (GANs)
* **Fichier :** `GenGAN.py`
* **Discriminator :** CNN classique (Convolution -> BatchNorm/InstanceNorm -> LeakyReLU).
* **Generator :** Reprend les architectures définies dans la partie Vanilla.
* **Modes d'entraînement :**
    * **BCE (Binary Cross Entropy) :** GAN classique. Loss combinée (Adversarial + L1 pour la reconstruction).
    * **WGAN-GP (Wasserstein GAN + Gradient Penalty) :** Amélioration significative de la stabilité. Utilise une pénalité de gradient au lieu du *weight clipping* et remplace la BatchNorm par l'**InstanceNorm** dans le critique.


### 4. Résultat 

| Architecture | Type d'entrée | Détail Technique | Train Loss | Val Loss  | Remarques |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla NN** | Vecteur (26) | MLP + Dropout | 0.0757 | 0.0769 | resultat un peu flou |
| **Vanilla NN** | Vecteur (26) | MLP + **SEBlock** (attention) | 0.0610 | 0.0659 | l attention aide bien le réseau |
| **Vanilla CNN** | Image | U-Net | 0.0635 | 0.0778 | mieux que le vecteur simple. |
| **Vanilla CNN** | Image | **U-Net + Self-Attention** | **0.0252** | **0.0556** | ** meilleur résultat** convergence rapide |

*(Note : les métriques pour les GAN ne sont pas reportées par oublie de noter ca lors des entrainements).*


### Analyse 

**1. Représentation de l'entree (Vecteur vs Image)**
Comme le papier de recherche, l utilisation d'une **image de squelette** est supérieure à l utilisation d'un vecteur de coordonnées.
* *Hypothèse :* Le réseau convolutionnel exploite mieux la cohérence spatiale des pixels d'une image là où un vecteur de coordonnées est une représentation trop abstraite qui rendrai plus compliqué de reconstruire une image

**2. Apport de l'Attention**
L'intégration de mécanismes d'attention (SEBlock pour les MLP, Self-Attention pour les CNN) ont permis d amélioré la convergence et la qualité visuelle

**3. Limites du GAN**
Contrairement aux conclusions du papier original, l'ajout du GAN n'a pas apporté d'amélioration visuelle significative par rapport au modèle du fichier Vanilla 
* *Hypothèse probable :* Je travaille sur une résolution faible (**64x64**). En effet, l intérêt principal d'un gan est de générer des détails haute fréquence pour justement éviter le flou. Avec cette taille des images, ces détail ne sont pas représentable ce qui fait que les GAN moins performant. 
Après j'ai très bien pu me tromper quelque part. 


---

## Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/linda-chouati/everybody-dance-now.git](https://github.com/linda-chouati/everybody-dance-now.git)
    cd everybody-dance-now
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *Dépendances principales : `torch`, `torchvision`, `numpy`, `opencv-python`, `mediapipe`, `streamlit`, `Pillow`.*

---

## Utilisation

### 1. Entraînement des modèles

On peut entrainer les différents modèles via les bon fichiers dédié 

* **Pour entraîner un réseau Vanilla :**
    Modifier `optSkeOrImage` (1 ou 2) dans le main
    ```bash
    python src/GenVanillaNN.py
    ```

* **Pour entraîner un GAN :**
    Modifier le `gan_mode` ("bce" ou "wgan-gp") et `optSkeOrImage` (1 ou 2) dans le main
    ```bash
    python src/GenGAN.py
    ```

### 2. Lancer la démo 

Pour visualiser le résultat :
1.  Ouvrir `src/DanceDemo.py`.
2.  Changer la variable `GEN_TYPE` pour choisir le modele à tester :
    ```python
    GEN_TYPE = 2  # 1=Nearest, 2=Vanilla Ske, 3=Vanilla Img, 6=WGAN Ske, 7=WGAN Img...
    ```
3.  Puis :
    ```bash
    python src/DanceDemo.py
    ```

### 3. Interface Web (Streamlit)

Une petite app a été developpée où l on peut choisir la video source et le type de générateur. 

* **Lancer en local :**
    ```bash
    streamlit run app.py
    ```
* **Lien app en ligne :** [Everybody Dance Now - Streamlit](https://everybody-dance-now.streamlit.app/)

