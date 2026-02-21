<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=1DB954&height=200&section=header&text=Spotify%20Tracks%20Dataset&fontSize=50&fontColor=ffffff&fontAlignY=38&desc=114K%20Songs%20%C2%B7%20114%20Genres%20%C2%B7%2020%20Audio%20Features&descAlignY=58&descSize=18" width="100%"/>

<br/>

[![Spotify](https://img.shields.io/badge/Spotify_Web_API-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://developer.spotify.com)
[![Python](https://img.shields.io/badge/Python_3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Kaggle](https://img.shields.io/badge/View_on_Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/saichaitanyareddyai)
[![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey?style=for-the-badge)](https://creativecommons.org/publicdomain/zero/1.0/)

<br/>

![Tracks](https://img.shields.io/badge/Tracks-114%2C000-1DB954?style=flat-square)
![Genres](https://img.shields.io/badge/Genres-114-20BEFF?style=flat-square)
![Artists](https://img.shields.io/badge/Artists-31%2C437-orange?style=flat-square)
![Columns](https://img.shields.io/badge/Columns-20-purple?style=flat-square)
![Missing](https://img.shields.io/badge/Missing_Values-None-brightgreen?style=flat-square)

</div>

<br/>

---

## 📖 Table of Contents

- [📌 Overview](#-overview)
- [📊 Dataset Statistics](#-dataset-statistics)
- [🧬 Feature Descriptions](#-feature-descriptions)
- [🚀 Quick Start](#-quick-start)
- [🎯 Use Cases](#-use-cases)
- [📈 Key Insights](#-key-insights)
- [🗂️ Repository Structure](#️-repository-structure)
- [🙌 Acknowledgements](#-acknowledgements)
- [👤 Author](#-author)
- [📜 License](#-license)

---

## 📌 Overview

> **A comprehensive collection of 114,000 Spotify tracks enriched with audio features — ideal for music analysis, genre classification, and building recommendation systems.**

This dataset was collected via the **Spotify Web API** and spans **114 unique music genres** with **31,437 unique artists**. Each track includes a rich set of audio features computed by Spotify's internal audio analysis engine.

Whether you are a **beginner** exploring EDA for the first time or an **advanced practitioner** building production ML models — this dataset provides meaningful real-world signals with zero missing values.

<br/>

> 🔗 **Also available on Kaggle →** [View Dataset](https://www.kaggle.com/saichaitanyareddyai)

---

## 📊 Dataset Statistics

<div align="center">

| Property | Value |
|:---:|:---:|
| 📁 **File Format** | CSV |
| 🎵 **Total Tracks** | 114,000 |
| 🎸 **Unique Genres** | 114 |
| 🧑‍🎤 **Unique Artists** | 31,437 |
| 📋 **Total Columns** | 20 |
| ❌ **Missing Values** | 0 |
| 📏 **Avg Popularity** | 33.2 / 100 |
| ⏱️ **Avg Duration** | ~3.8 minutes |
| 🎹 **Avg Tempo** | 122 BPM |
| 🎼 **Major Key Tracks** | 63.8% |
| 💃 **Avg Danceability** | 0.567 |
| ⚡ **Avg Energy** | 0.641 |

</div>

---

## 🧬 Feature Descriptions

<details>
<summary><b>🔍 Click to expand full feature table</b></summary>

<br/>

| Column | Type | Range | Description |
|---|---|---|---|
| `track_id` | String | — | Unique Spotify URI identifier for the track |
| `artists` | String | — | Performing artist name(s) |
| `album_name` | String | — | Album the track belongs to |
| `track_name` | String | — | Title of the track |
| `popularity` | Integer | 0 – 100 | Spotify popularity score (higher = more popular) |
| `duration_ms` | Integer | 0 – 5.2M | Track duration in milliseconds |
| `explicit` | Boolean | True/False | Whether the track has explicit lyrics |
| `danceability` | Float | 0.0 – 1.0 | How suitable a track is for dancing |
| `energy` | Float | 0.0 – 1.0 | Perceptual measure of intensity and activity |
| `key` | Integer | 0 – 11 | Musical key (0=C, 1=C♯, 2=D … 11=B) |
| `loudness` | Float | -49.5 – 4.5 | Overall loudness in decibels (dB) |
| `mode` | Integer | 0 or 1 | Modality — 1 = Major, 0 = Minor |
| `speechiness` | Float | 0.0 – 1.0 | Presence of spoken words in the track |
| `acousticness` | Float | 0.0 – 1.0 | Confidence the track is acoustic |
| `instrumentalness` | Float | 0.0 – 1.0 | Predicts whether a track has no vocals |
| `liveness` | Float | 0.0 – 1.0 | Detects presence of a live audience |
| `valence` | Float | 0.0 – 1.0 | Musical positiveness conveyed by the track |
| `tempo` | Float | 0 – 243 | Estimated tempo in beats per minute (BPM) |
| `time_signature` | Integer | 0 – 5 | Estimated time signature of the track |
| `track_genre` | String | 114 classes | Genre label assigned to the track |

</details>

---

## 🚀 Quick Start

### Step 1 — Clone the repository
```bash
git clone https://github.com/saichaitanyareddyai/spotify-tracks-dataset.git
cd spotify-tracks-dataset
```

### Step 2 — Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3 — Load and explore
```python
import pandas as pd

# Load dataset
df = pd.read_csv('spotify-tracks-dataset-clean.csv')

# Basic info
print(f"Shape      : {df.shape}")
print(f"Genres     : {df['track_genre'].nunique()}")
print(f"Artists    : {df['artists'].nunique()}")
print(f"Null values: {df.isnull().sum().sum()}")

# Preview
df.head()
```

### Step 4 — Quick EDA example
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Top 10 genres by average popularity
top_genres = (df.groupby('track_genre')['popularity']
                .mean()
                .sort_values(ascending=False)
                .head(10))

top_genres.plot(kind='bar', color='#1DB954', edgecolor='white')
plt.title('Top 10 Genres by Average Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Average Popularity Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_genres.png', dpi=150)
plt.show()
```

---

## 🎯 Use Cases

| # | Use Case | Difficulty | Key Techniques |
|---|---|:---:|---|
| 1 | 🔍 **Exploratory Data Analysis** | `Beginner` | pandas, seaborn, matplotlib |
| 2 | 🎸 **Genre Classification** | `Intermediate` | Random Forest, XGBoost, SVM |
| 3 | 📈 **Popularity Prediction** | `Intermediate` | Linear Regression, Gradient Boosting |
| 4 | 🎧 **Music Recommendation System** | `Advanced` | K-Means, Cosine Similarity, Collaborative Filtering |
| 5 | 🧪 **Feature Engineering Practice** | `Intermediate` | Encoding, Binning, Normalization |
| 6 | 🧠 **Deep Learning Classification** | `Advanced` | Neural Networks, Embeddings, PyTorch |
| 7 | 📊 **Audio Feature Correlation Study** | `Beginner` | Heatmaps, Pairplots, Correlation Matrix |
| 8 | 🌍 **Genre Clustering** | `Intermediate` | DBSCAN, t-SNE, UMAP |

---

## 📈 Key Insights
╔══════════════════════════════════════════════════════════╗
║              DATASET HIGHLIGHTS AT A GLANCE              ║
╠══════════════════════════════════════════════════════════╣
║  🎵  Avg Popularity     →   33.2 / 100  (mostly niche)   ║
║  ⚡  Avg Energy         →   0.64        (leans energetic) ║
║  💃  Avg Danceability   →   0.57        (moderately danceable) ║
║  😊  Avg Valence        →   0.47        (slightly melancholic) ║
║  🎹  Avg Tempo          →   122 BPM     (pop/dance range)  ║
║  🎼  Major Key Tracks   →   63.8%       (majority upbeat)  ║
║  🎤  Avg Speechiness    →   0.085       (mostly musical)   ║
║  🎸  Avg Acousticness   →   0.315       (mix of both)      ║
╚══════════════════════════════════════════════════
