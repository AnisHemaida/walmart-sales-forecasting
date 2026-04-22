<div align="center">

<img src="images/Page de garde.png" alt="Page de garde" width="100%" style="border-radius: 12px;" />

<br/>

# <span style="color:#0071CE">🛒 Système Intelligent de Prévision des Ventes</span>
## <span style="color:#FFC220">En Grande Distribution — Walmart</span>

<br/>

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-6.6.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br/>

[![Cours](https://img.shields.io/badge/Cours-IFM30546--0010--H2026-0071CE?style=flat-square)](.)
[![Session](https://img.shields.io/badge/Session-4-FFC220?style=flat-square)](.)
[![Licence](https://img.shields.io/badge/Licence-Académique-green?style=flat-square)](.)
[![R² Score](https://img.shields.io/badge/R²-98.55%25-brightgreen?style=flat-square)](.)

</div>

---

<div align="center">

## <span style="color:#0071CE">📋 Table des matières</span>

</div>

> - [🎯 À propos du projet](#-à-propos-du-projet)
> - [👥 Équipe](#-équipe)
> - [🏗️ Architecture du projet](#-architecture-du-projet)
> - [📊 Données](#-données)
> - [🤖 Pipeline Machine Learning](#-pipeline-machine-learning)
> - [🌐 Application Web Flask](#-application-web-flask)
> - [📈 Performances du modèle](#-performances-du-modèle)
> - [🚀 Installation et démarrage](#-installation-et-démarrage)
> - [🧪 Tests et prédictions](#-tests-et-prédictions)
> - [📁 Structure des fichiers](#-structure-des-fichiers)
> - [🛠️ Stack technique](#-stack-technique)

---

## 🎯 À propos du projet

<table>
<tr>
<td width="60%">

Ce projet développe un **système intelligent de prévision des ventes** pour les magasins Walmart, combinant des techniques avancées de **Machine Learning** et un tableau de bord interactif en temps réel.

Le système permet de :

- 🔮 **Prédire** les ventes hebdomadaires par magasin et département
- 📦 **Optimiser** la gestion des stocks (modèle EOQ + stocks de sécurité)
- 💰 **Simuler** l'impact des promotions et markdowns
- 📉 **Analyser** les tendances temporelles et saisonnières
- 🏪 **Comparer** les performances entre types de magasins (A, B, C)

</td>
<td width="40%" align="center">

### 🏆 Résultats clés

| Métrique | Valeur |
|----------|--------|
| 📊 R² Score | **98,55 %** |
| 💵 WMAE | **1 332 $** |
| 🏪 Magasins | **45** |
| 🗂️ Départements | **81** |
| 📅 Données | **421 570** lignes |
| 📆 Période | **2010 – 2012** |

</td>
</tr>
</table>

---

## 👥 Équipe

<div align="center">

| 👤 Membre |
|-----------|
| **Anis Hemaida** |
| **Wassila Ennouar** |
| **Salsabil Reguragui** | 
| **Mohamed Lamine Zoutat** | 
| **Farid Bandoui** | 
| **Abdennour Kerrouch** | 

> **Établissement :** La Cité — Collège d'arts appliqués et de technologie  
> **Programme :** Sciences des données — Session 4

</div>

---

## 🏗️ Architecture du projet

```
📦 Système intelligent de prévision des ventes/
│
├── 📂 Data/                          # Données brutes et préparées
│   ├── 📄 train.csv                  # Données d'entraînement (421 570 lignes)
│   ├── 📄 test.csv                   # Données de test (115 065 lignes)
│   ├── 📄 features.csv               # Variables économiques et promotionnelles
│   ├── 📄 stores.csv                 # Métadonnées des 45 magasins
│   └── 📂 prepared/
│       └── 📄 df_train.parquet       # Données préparées (format optimisé)
│
├── 📂 models/                        # Modèles ML sérialisés
│   ├── 🤖 model.pkl                  # Modèle XGBoost entraîné (5 MB)
│   ├── 🔧 scaler.pkl                 # StandardScaler
│   ├── 📋 columns.pkl                # Noms des features
│   ├── 🗺️ store_mapping.pkl          # Encodage des magasins
│   └── 🗺️ dept_mapping.pkl           # Encodage des départements
│
├── 📂 notebooks/                     # Jupyter Notebooks (phases EPIC)
│   ├── 📓 SIPV_Walmart_EPIC1.ipynb   # EPIC 1 : Compréhension et préparation
│   ├── 📓 SIPV_Walmart_EPIC2.ipynb   # EPIC 2 : Entraînement et sélection du modèle
│   └── 📓 SIPV_Walmart_EPIC3.ipynb   # EPIC 3 : Dashboard et prédictions
│
├── 📂 flask_app/                     # Application Web
│   ├── 🐍 app.py                     # API REST Flask (972 lignes)
│   ├── 🖥️ Lancer_Interface.bat       # Lanceur Windows
│   ├── 📂 templates/
│   │   └── 🌐 index.html             # Dashboard React + Plotly
│   └── 📂 static/
│       ├── 🎨 style.css              # Styles CSS
│       └── 🖼️ [images Walmart]       # Logos et en-têtes
│
├── 📂 reports/                       # Rapports et exports
│   ├── 📄 Rapport_sipv_walmart.pdf   # Rapport final (2.2 MB)
│   ├── 📊 historique_walmart.csv     # Historique enrichi (38 MB)
│   └── 📊 previsions_store.csv       # Prévisions 2012 vs réel
│
├── 📂 images/                        # Images du projet
├── 📄 test_batch_prevision.csv       # Fichier test prédictions par lot
└── 📄 README.md                      # Ce fichier
```

---

## 📊 Données

### Sources de données

<table>
<thead>
<tr>
<th>📁 Fichier</th>
<th>📏 Taille</th>
<th>📝 Description</th>
<th>🔑 Colonnes clés</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>train.csv</code></td>
<td>13 MB</td>
<td>Ventes hebdomadaires d'entraînement</td>
<td>Store, Dept, Date, Weekly_Sales, IsHoliday</td>
</tr>
<tr>
<td><code>test.csv</code></td>
<td>2.5 MB</td>
<td>Ensemble de test (sans étiquettes)</td>
<td>Store, Dept, Date, IsHoliday</td>
</tr>
<tr>
<td><code>features.csv</code></td>
<td>579 KB</td>
<td>Variables économiques et promotions</td>
<td>Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment</td>
</tr>
<tr>
<td><code>stores.csv</code></td>
<td>532 B</td>
<td>Métadonnées des 45 magasins</td>
<td>Store, Type (A/B/C), Size</td>
</tr>
</tbody>
</table>

### Aperçu des données d'entraînement

```
Période     : 5 février 2010 → 26 octobre 2012 (~160 semaines)
Magasins    : 45 (Types A, B, C)
Départements: jusqu'à 81 par magasin
Total lignes: 421 570 observations
```

### Variables économiques (features.csv)

| Variable | Description |
|----------|-------------|
| `Temperature` | Température moyenne hebdomadaire (°F) |
| `Fuel_Price` | Prix du carburant par région |
| `MarkDown1-5` | Indicateurs de promotions (réductions) |
| `CPI` | Indice des prix à la consommation |
| `Unemployment` | Taux de chômage hebdomadaire |
| `IsHoliday` | Indicateur semaine de congé (0/1) |

---

## 🤖 Pipeline Machine Learning

### Vue d'ensemble du pipeline

```
Données brutes
     │
     ▼
┌─────────────────┐
│  Prétraitement  │  ← Fusion train + features + stores
│  & Nettoyage    │  ← Remplissage valeurs manquantes (MarkDown → 0)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │  ← Temporel (Year, Month, Week, Quarter)
│   Engineering   │  ← Cyclique (sin/cos pour saisonnalité)
│                 │  ← Lags (1, 4, 12, 52 semaines)
│                 │  ← Moyennes mobiles (MA_4, MA_12)
│                 │  ← Encodage ordinal (Store Type A=3, B=2, C=1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Normalisation │  ← StandardScaler (Temperature, Fuel_Price, CPI,
│                 │    Unemployment, MarkDowns, Size)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    XGBoost      │  ← Régression sur Weekly_Sales
│   Regressor     │  ← R² = 98.55%  |  WMAE = 1 264 $
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prédictions   │  ← Unitaires / Par horizon / Par lot
│   & Alertes     │  ← Gestion des stocks (EOQ + Stocks de sécurité)
└─────────────────┘
```

### Features engineered

<table>
<tr>
<th>🕐 Temporelles</th>
<th>📦 Lags & MA</th>
<th>🏷️ Promotions</th>
<th>🏪 Magasin</th>
<th>💹 Économiques</th>
</tr>
<tr>
<td>

- Year
- Month
- Week
- Quarter
- DayOfYear
- Month_sin / cos
- Week_sin / cos

</td>
<td>

- Lag_1
- Lag_4
- Lag_12
- Lag_52
- MA_4
- MA_12

</td>
<td>

- MarkDown1-5
- MarkDown_total
- HasMarkDown

</td>
<td>

- Store (encodé)
- Dept (encodé)
- Type_ord (A=3, B=2, C=1)
- Size

</td>
<td>

- Temperature
- Fuel_Price
- CPI
- Unemployment
- IsHoliday

</td>
</tr>
</table>

### Gestion des stocks

| Formule | Description |
|---------|-------------|
| `EOQ = √(2 × D × S / H)` | Quantité économique de commande |
| `Safety Stock = Z × σ × √L` | Stock de sécurité |
| `CV = σ / μ` | Coefficient de variation (indicateur de risque) |

**Niveaux d'alerte :**

| 🚦 Niveau | CV | Signification |
|----------|-----|---------------|
| 🔴 CRITIQUE | > 0.40 | Risque de rupture élevé |
| 🟠 HAUTE | > 0.25 | Surveillance rapprochée |
| 🟡 MODERÉE | > 0.15 | Attention recommandée |
| 🟢 OK | ≤ 0.15 | Stock stable |

---

## 🌐 Application Web Flask

### Pages du tableau de bord

| # | Page | Description |
|---|------|-------------|
| 1 | **Vue d'ensemble** | KPIs globaux, analyse Pareto, top départements |
| 2 | **Tendances temporelles** | Évolution mensuelle, saisonnalité, repères fériés |
| 3 | **Types de magasins** | Comparaison A/B/C, boîtes à moustaches, ventes moyennes |
| 4 | **Analyse par département** | Top 15, Flop 10, tendances saisonnières |
| 5 | **Impact des promotions** | Lift par MarkDown, ROI simulé |
| 6 | **Jours fériés** | Super Bowl, Thanksgiving, Noël — impact par type |
| 7 | **Variables économiques** | Température, carburant, CPI, chômage |
| 8 | **Corrélations & Features** | Importance SHAP, corrélations ML |
| 9 | **Recommandations** | Actions prioritaires classées par impact |
| 10 | **Prédictions** | Interface interactive de prévision |

### API REST — Endpoints principaux

```http
# Analytics
GET  /api/analytics/overview       → KPIs globaux + analyse Pareto
GET  /api/analytics/temporal       → Tendances mensuelles & saisonnalité
GET  /api/analytics/stores         → Performance par type (A/B/C)
GET  /api/analytics/departments    → Top/Flop départements
GET  /api/analytics/promotions     → Lift des markdowns
GET  /api/analytics/holidays       → Impact des jours fériés
GET  /api/analytics/economic       → Variables économiques
GET  /api/analytics/correlations   → Corrélations des features

# Prédictions
POST /api/predict                  → Prédiction unitaire
POST /api/predict/horizon          → Prévision sur 4 semaines
POST /api/simulate_promo           → Simulation ROI promotion
POST /api/predict/batch            → Prédictions par lot (CSV/Excel)

# Stocks
GET  /api/stock                    → EOQ + stocks de sécurité + alertes

# Données
GET  /api/kpis                     → KPIs résumés
GET  /api/historique               → Historique agrégé
GET  /api/top_stores               → Top 10 magasins
GET  /export/csv                   → Export CSV
```

---

## 📈 Performances du modèle

<div align="center">

| 🎯 Métrique | 📊 Valeur | 💡 Interprétation |
|------------|----------|------------------|
| **R² Score** | **98,55 %** | Le modèle explique 98,55 % de la variance des ventes |
| **WMAE** | **1 264 $** | Erreur absolue pondérée moyenne par semaine/département |
| **Données entraînement** | **421 570** | Observations couvrant 3 ans |
| **Validation** | Août–Oct 2012 | Prédictions vs ventes réelles |

</div>

### Facteurs d'impact identifiés

| 🏷️ Promotion | 📈 Lift moyen |
|-------------|--------------|
| MarkDown1 | **+18,5 %** |
| MarkDown2 | **+12,3 %** |
| Thanksgiving | **+28,5 %** |
| Noël (Christmas) | **+21,3 %** |
| Super Bowl | Impact variable selon type de magasin |

---

## 🚀 Installation et démarrage

### Prérequis

- Python **3.13+**
- pip / virtualenv

### 1. Cloner le dépôt

```bash
git clone https://github.com/AnisHemaida/walmart-sales-forecasting.git
cd walmart-sales-forecasting
```

### 2. Installer les dépendances

```bash
pip install flask pandas numpy xgboost scikit-learn plotly
```

> **Note :** Aucun fichier `requirements.txt` n'est inclus — installer les packages ci-dessus suffit pour lancer l'application.

### 3. Lancer l'application

**Option A — Terminal :**
```bash
cd flask_app
python app.py
```

**Option B — Lanceur Windows :**
```
Double-cliquer sur : flask_app/Lancer_Interface.bat
```

### 4. Accéder au tableau de bord

```
http://localhost:5000
```

---

## 🧪 Tests et prédictions

### Prédiction unitaire

```python
import requests

response = requests.post("http://localhost:5000/api/predict", json={
    "store": 1,
    "dept": 1,
    "date": "2012-10-05",
    "markdown": 1000
})
print(response.json())
# → {"prediction": 24532.75, "store": 1, "dept": 1, ...}
```

### Prévision sur horizon (4 semaines)

```python
response = requests.post("http://localhost:5000/api/predict/horizon", json={
    "store": 1,
    "dept": 1,
    "start_date": "2012-10-01"
})
```

### Simulation de promotion

```python
response = requests.post("http://localhost:5000/api/simulate_promo", json={
    "store": 1,
    "dept": 1,
    "md1": 5000,
    "md2": 3000,
    "cout_promo": 3000,
    "marge": 0.30
})
# → ROI, ventes supplémentaires, profit net estimé
```

### Prédictions par lot (CSV)

Utiliser le fichier `test_batch_prevision.csv` comme exemple d'entrée pour l'endpoint `/api/predict/batch`.

Colonnes attendues :

```
Store, Dept, Date, [Temperature, Fuel_Price, CPI, Unemployment, IsHoliday, MarkDown1-5]
```

### Recommandations de stock

```
GET http://localhost:5000/api/stock?store=1&dept=1&S=50&H=0.20&L=1&Z=1.65
```

---

## 📁 Structure des fichiers

| 📄 Fichier | 📏 Taille | 🎯 Usage |
|-----------|---------|---------|
| `Data/train.csv` | 13 MB | Entraînement du modèle |
| `Data/features.csv` | 579 KB | Variables économiques et promotions |
| `Data/stores.csv` | 532 B | Métadonnées des magasins |
| `models/model.pkl` | 5.0 MB | Modèle XGBoost (binaire) |
| `models/scaler.pkl` | 899 B | StandardScaler |
| `flask_app/app.py` | ~22 KB | API Flask (972 lignes) |
| `flask_app/templates/index.html` | ~46 KB | Dashboard React |
| `flask_app/static/style.css` | 8.4 KB | Styles CSS |
| `reports/Rapport_sipv_walmart.pdf` | 2.2 MB | Rapport final |
| `reports/historique_walmart.csv` | 38 MB | Historique enrichi |
| `reports/previsions_store.csv` | 1.6 MB | Prévisions 2012 vs réel |
| `test_batch_prevision.csv` | 8.1 KB | Exemple prédictions par lot |

---

## 🛠️ Stack technique

<div align="center">

### Backend

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-FF6600?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.3.1-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Frontend

![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly.js-2.26-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Babel](https://img.shields.io/badge/Babel-JSX-F9DC3E?style=for-the-badge&logo=babel&logoColor=black)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### Outils & Environnement

![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-Version_Control-F05032?style=for-the-badge&logo=git&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-11-0078D6?style=for-the-badge&logo=windows&logoColor=white)

</div>

---

<div align="center">

## 📒 Notebooks — Phases EPIC

| Phase | Notebook | Contenu |
|-------|----------|---------|
| **EPIC 1** | `SIPV_Walmart_EPIC1.ipynb` | Compréhension du contexte, exploration des données, analyse qualité |
| **EPIC 2** | `SIPV_Walmart_EPIC2.ipynb` | Feature engineering, entraînement des modèles, évaluation, SHAP |
| **EPIC 3** | `SIPV_Walmart_EPIC3.ipynb` | Dashboard, prédictions, simulateur promo, recommandations stocks |

</div>

---

<div align="center">

---

*Projet réalisé dans le cadre du cours **Sciences des données (IFM30546-0010-H2026)***  
*La Cité — Collège d'arts appliqués et de technologie — Session H2026*

---

</div>
