# Nash-DQN: Multi-Agent Deep Q-Learning for Optimal Trade Execution (Casgrain et al., 2022)

> **Cadre académique** : McMaster University (Canada) — MATH 4FM3 (Financial Mathematics)  
> **Auteur** : Mohamed Nourou Dine CISSE  
> **Domaine** : `Quantitative Finance, Multi-Agent Reinforcement Learning & Stochastic Games`  
> **Stack Technique** : `Python` `PyTorch` `Deep Q-Networks (DQN)` `Multi-Agent RL` `Nash Equilibrium` `Almgren-Chriss Framework` `Brownian Motion`

---

## 📌 Présentation Réelle du Projet (Honest Overview)

Ce projet de recherche quantitative a été réalisé dans le cadre de mon semestre d'échange académique à **McMaster University** (Ontario, Canada). Il implémente et évalue l'algorithme **Nash-DQN** proposé par **Casgrain, Ning & Jaimungal (2022)** pour l'exécution optimale d'ordres en finance de marché dans un cadre de jeu différentiel stochastique à $N$ agents compétitifs.

Dans ce cadre :
- Chaque agent doit liquider un inventaire donné avant un horizon temporel $T$.
- Les agents interagissent à travers un prix de marché unique soumis à un **impact temporaire** (pénalisant les transactions instantanées trop agressives) et un **impact permanent** (qui déplace durablement le prix contre l'ensemble des vendeurs).
- L'algorithme Nash-DQN entraîne des réseaux de neurones profonds couplés à une couche invariante par permutation pour converger vers la politique d'action conjointe formant un **Équilibre de Nash**.

---

## 🏗️ Structure du Répertoire

```text
09_McMaster_Nash_DQN_MultiAgent_Trading/
├── README.md                      # Documentation complète
├── visuals/
│   ├── brownian_paths.png         # Trajectoires browniennes simulées du prix d'actif
│   └── all_agents_rewards.png     # Courbes de convergence des récompenses des agents
├── reports/
│   ├── Rapport_Nash_DQN_Optimal_Trading.pdf # Mémoire de recherche quantitatif complet (34 pages)
│   ├── Project_Roadmap.pdf        # Feuille de route du projet de recherche
│   └── Rapport_source.tex         # Code source LaTeX du rapport
└── src/
    ├── environment.py             # Simulateur de marché stochastique à N agents (impact permanent/temporaire)
    ├── model.py                   # Architectures PyTorch : PermutationInvariantLayer, ValueNet, AdvantageNet, NashDQN
    ├── buffer.py                  # Replay buffer vectorisé pour l'expérience multi-agents
    ├── train.py                   # Script d'entraînement avec ordonnancement du bruit et visualisations
    ├── new_train.py               # Entraînement multi-graines (multi-seed) et lissage par moyenne mobile
    ├── Project.ipynb              # Notebook interactif complet de démonstration
    ├── concavity_check.png        # Diagnostic de concavité de la fonction d'avantage
    └── tests/                     # Tests unitaires des composants mathématiques
        ├── test_environment.py    # Validation des dynamiques d'inventaire et de prix
        ├── test_buffer.py         # Validation des formes de tenseurs et échantillonnage
        ├── test_TWAP.py           # Benchmark de référence TWAP (Time-Weighted Average Price)
        ├── test_FP.py             # Validation du calcul de point fixe / équilibre
        └── NN_test.py             # Vérification de l'invariance par étiquetage des agents
```

---

## 📐 Formulation Mathématique du Simulateur

La dynamique du prix $S_t$ obéit à une diffusion d'Itô soumise à l'impact permanent moyen des $N$ traders :
$$dS_t = \bar{a} \left(\frac{1}{N} \sum_{j=1}^N \nu_t^j\right) dt + \sigma dW_t$$

Chaque agent $i$ contrôle sa vitesse de trading $\nu_t^i = \dot{q}_t^i$ et maximise sa fonctionnelle d'utilité terminale pénalisée par le risque d'inventaire :
$$J^i(\nu^1, \dots, \nu^N) = \mathbb{E}\left[\int_0^T \left(S_t \nu_t^i - k (\nu_t^i)^2 - \phi (q_t^i)^2\right) dt + q_T^i (S_T - \alpha q_T^i)\right]$$

L'architecture neuronale décompose la Q-valeur selon une formulation d'avantage quadratique garantissant l'existence et l'unicité de la meilleure réponse conjointe.

---

## 🚀 Guide d'Exécution & Tests

### Prérequis
```bash
pip install torch numpy matplotlib jupyter
```

### 1. Lancement des tests unitaires
```bash
cd src
python -m pytest tests/
# ou individuellement :
python tests/test_environment.py
python tests/NN_test.py
```

### 2. Entraînement du modèle
```bash
cd src
python train.py
# ou pour l'entraînement multi-seed :
python new_train.py
```

### 3. Notebook interactif
```bash
cd src
jupyter notebook Project.ipynb
```

---

## 📄 Preuves & Publications Incluses
- 📄 **Thèse de Recherche Quantitative (34 pages)** : [`reports/Rapport_Nash_DQN_Optimal_Trading.pdf`](reports/Rapport_Nash_DQN_Optimal_Trading.pdf)
- 📊 **Courbes de Convergence & Visualisations** : Dossier [`visuals/`](visuals/)

---

## 📬 Profil de l'Auteur
- **Mohamed Nourou Dine CISSE**
- **ENSIIE** & **Université Paris-Saclay** — Double diplôme, Master 2 Intelligence Artificielle (M2IA)
- Échange académique : **McMaster University** (Canada)
- **GitHub** : [@nourou-dine-cisse](https://github.com/nourou-dine-cisse)
