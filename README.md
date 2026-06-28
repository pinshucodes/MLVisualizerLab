<p align="center">
  <img src="https://mlvisualizerlabfullstackwebapp.streamlit.app/" alt="ML Visualizer Lab" height="60"/>
</p>

<p align="center">
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://github.com/pinshucodes/MLVisualizerLab/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License"/></a>
  <a href="https://github.com/pinshucodes/MLVisualizerLab/stargazers"><img src="https://img.shields.io/github/stars/pinshucodes/MLVisualizerLab?style=for-the-badge&color=F59E0B&logo=github" alt="GitHub Stars"/></a>
</p>

<p align="center">
  <strong>🌐 Live App: <a href="https://mlvisualizerlabfullstackwebapp.streamlit.app/">mlvisualizerlabfullstackwebapp.streamlit.app</a></strong>
</p>

<p align="center">
  <strong>An interactive machine learning exploration platform for education, experimentation, and visualization.</strong>
</p>

<p align="center">
  Train models · Visualize results · Explain predictions · Compare algorithms<br/>
  <em>All from your browser — no code required.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-supported-algorithms">Algorithms</a> •
  <a href="#-usage-guide">Usage Guide</a> •
  <a href="#%EF%B8%8F-deployment">Deploy</a> •
  <a href="#-contributors">Contributors</a>
</p>

---

| ![Comparison](screenshots/comparison.png) |
| **Dimensionality Reduction** | ![Dimensionality](screenshots/dimensionality.png) |
| **Clustering** | ![Clustering](screenshots/clustering.png) |

</details>

## ✨ Features

ML Visualizer Lab is a **full-pipeline ML workbench** packed into a single Streamlit app. Here's everything you can do:

| Category | Feature | Description |
|:---------|:--------|:------------|
| 📁 **Data** | Dataset Loading | Upload CSV/Excel files or choose from built-in Scikit-learn datasets (Iris, Wine, Breast Cancer, Diabetes, etc.) |
| 📊 **Data** | Dataset Overview | Interactive statistics, shape, dtypes, missing values, and distribution plots |
| 🔧 **Data** | Preprocessing | Handle missing values, encode categoricals, scale features (Standard, MinMax, Robust), and split train/test |
| 🎯 **Modeling** | Model Training | Train classification or regression models with configurable hyperparameters via sidebar controls |
| 📈 **Modeling** | Evaluation | Confusion matrix, ROC/AUC curves, precision-recall curves, classification report, and regression metrics (R², MAE, MSE, RMSE) |
| 🎨 **Modeling** | Visualizations | Interactive Plotly charts — scatter plots, histograms, box plots, and correlation heatmaps |
| 🧠 **Advanced** | Explainability | SHAP summary plots, force plots, waterfall plots, and feature importance rankings |
| ⚔️ **Advanced** | Model Comparison | Train and compare multiple algorithms side-by-side with metric tables and radar charts |
| 📉 **Advanced** | Dimensionality Reduction | PCA, t-SNE, and UMAP visualizations for high-dimensional data exploration |
| 🔮 **Advanced** | Clustering | K-Means and DBSCAN with elbow plots, silhouette analysis, and 2D/3D cluster visualizations |
| 📐 **Analysis** | Cross Validation | K-Fold, Stratified K-Fold cross-validation with per-fold performance breakdown |
| 📚 **Analysis** | Learning Curves | Diagnose bias/variance trade-offs with training and validation learning curves |
| ⚡ **Analysis** | Hyperparameter Optimization | Grid Search and Randomized Search with result visualizations |
| 🧪 **Analysis** | Experiment Tracking | Log experiments with parameters, metrics, and timestamps for reproducible comparison |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** installed on your machine
- **pip** or a virtual environment manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pinshucodes/MLVisualizerLab.git
cd MLVisualizerLab

# 2. Create a virtual environment (recommended)
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will launch at **http://localhost:8501** in your default browser. 🎉

---

## 📖 Usage Guide

Follow these steps for a complete ML workflow — no coding needed:

### Step 1 — Load Your Data

Navigate to **🏠 Home** and either:
- **Upload** a CSV or Excel file from your machine, or
- **Select** a built-in dataset (Iris, Wine, Breast Cancer, Diabetes, etc.)

### Step 2 — Explore the Dataset

Go to **📊 Dataset Overview** to inspect:
- Shape, columns, data types, and summary statistics
- Missing value analysis
- Feature distribution plots

### Step 3 — Preprocess

Head to **🔧 Preprocessing** to:
- Handle missing values (drop, fill with mean/median/mode)
- Encode categorical variables (Label / One-Hot encoding)
- Scale numeric features (Standard, MinMax, or Robust scaler)
- Select features and set the target column
- Configure the train/test split ratio

### Step 4 — Train a Model

On the **🎯 Training** page:
1. Choose a task type — **Classification** or **Regression**
2. Select an algorithm from the dropdown
3. Tune hyperparameters using interactive sidebar widgets
4. Click **Train** and watch the results

### Step 5 — Evaluate

Visit **📈 Evaluation** for detailed performance metrics:
- **Classification:** Confusion matrix, ROC curve, Precision-Recall curve, classification report
- **Regression:** R² score, MAE, MSE, RMSE, residual plots

### Step 6 — Explain & Analyze

Dive deeper with the **🧠 Advanced** and **📐 Analysis** pages:
- **Explainability** — Understand *why* the model makes its predictions using SHAP
- **Model Comparison** — Pit multiple algorithms against each other
- **Dimensionality Reduction** — Visualize data in 2D/3D with PCA, t-SNE, or UMAP
- **Clustering** — Discover natural groupings in your data
- **Cross Validation** — Get robust performance estimates
- **Learning Curves** — Diagnose overfitting / underfitting
- **Hyperparameter Optimization** — Automatically search for the best config

---

## 🤖 Supported Algorithms

### Classification

| Algorithm | Library | Key Hyperparameters |
|:----------|:--------|:--------------------|
| K-Nearest Neighbors | Scikit-learn | `n_neighbors`, `weights`, `metric` |
| Logistic Regression | Scikit-learn | `C`, `penalty`, `solver`, `max_iter` |
| Support Vector Machine | Scikit-learn | `C`, `kernel`, `gamma` |
| Decision Tree | Scikit-learn | `max_depth`, `criterion`, `min_samples_split` |
| Random Forest | Scikit-learn | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting | Scikit-learn | `n_estimators`, `learning_rate`, `max_depth` |
| XGBoost | XGBoost | `n_estimators`, `learning_rate`, `max_depth` |
| LightGBM | LightGBM | `n_estimators`, `learning_rate`, `max_depth` |

### Regression

| Algorithm | Library | Key Hyperparameters |
|:----------|:--------|:--------------------|
| Linear Regression | Scikit-learn | — |
| Ridge Regression | Scikit-learn | `alpha` |
| Lasso Regression | Scikit-learn | `alpha` |
| Decision Tree Regressor | Scikit-learn | `max_depth`, `min_samples_split` |
| Random Forest Regressor | Scikit-learn | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting Regressor | Scikit-learn | `n_estimators`, `learning_rate`, `max_depth` |
| XGBoost Regressor | XGBoost | `n_estimators`, `learning_rate`, `max_depth` |
| LightGBM Regressor | LightGBM | `n_estimators`, `learning_rate`, `max_depth` |

### Clustering

| Algorithm | Library | Key Hyperparameters |
|:----------|:--------|:--------------------|
| K-Means | Scikit-learn | `n_clusters`, `n_init`, `random_state` |
| DBSCAN | Scikit-learn | `eps`, `min_samples` |

---

## 📁 Project Structure

```
MLVisualizerLab/
│
├── app.py                          # 🚀 Main Streamlit entry point & navigation
├── requirements.txt                # 📦 Python dependencies
├── packages.txt                    # 📦 System-level packages (for deployment)
│
├── .streamlit/
│   └── config.toml                 # ⚙️  Streamlit theme & settings
│
├── pages/                          # 📄 Multi-page Streamlit modules
│   ├── home.py                     #    🏠 Landing page & dataset selector
│   ├── dataset.py                  #    📊 Dataset overview & statistics
│   ├── preprocessing.py            #    🔧 Data cleaning & feature engineering
│   ├── training.py                 #    🎯 Model training interface
│   ├── evaluation.py               #    📈 Performance metrics & plots
│   ├── visualizations.py           #    🎨 Interactive data visualizations
│   ├── explainability.py           #    🧠 SHAP explainability dashboard
│   ├── comparison.py               #    ⚔️  Side-by-side model comparison
│   ├── dimensionality.py           #    📉 PCA / t-SNE / UMAP reduction
│   ├── clustering.py               #    🔮 K-Means & DBSCAN clustering
│   ├── cross_validation.py         #    📐 K-Fold cross-validation
│   ├── learning_curves.py          #    📚 Bias/variance diagnostics
│   ├── hyperparameter_opt.py       #    ⚡ Grid & Random search
│   └── experiments.py              #    🧪 Experiment logging & tracking
│
├── models/                         # 🤖 Algorithm registries & factories
│   ├── __init__.py
│   ├── classifiers.py              #    Classification algorithms
│   ├── regressors.py               #    Regression algorithms
│   └── clustering.py               #    Clustering algorithms
│
├── utils/                          # 🛠️  Shared utility modules
│   ├── __init__.py
│   ├── data_loader.py              #    Dataset loading & sample data
│   ├── preprocessing.py            #    Encoding, scaling, splitting
│   ├── metrics.py                  #    Metric computation helpers
│   ├── plots.py                    #    Plotly chart builders
│   └── export.py                   #    Model & results export
│
└── data/
    └── sample_iris.csv             # 🌸 Sample dataset for quick start
```

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended)

The easiest way to deploy — completely free:

1. **Push** your code to a public GitHub repository
2. Go to [**share.streamlit.io**](https://share.streamlit.io)
3. Click **"New app"** and connect your repository
4. Set the following:
   - **Repository:** `pinshucodes/MLVisualizerLab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy** 🚀

> [!TIP]
> The `packages.txt` file is automatically picked up by Streamlit Cloud for system-level dependencies.

### Other Platforms

<details>
<summary><strong>Docker</strong></summary>

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ml-visualizer-lab .
docker run -p 8501:8501 ml-visualizer-lab
```

</details>

<details>
<summary><strong>Heroku</strong></summary>

Create a `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

Then deploy:

```bash
heroku create ml-visualizer-lab
git push heroku main
```

</details>

<details>
<summary><strong>Railway / Render</strong></summary>

Both platforms auto-detect Python apps. Set the **start command** to:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

</details>

---

## 🛠️ Tech Stack

| Technology | Purpose | Version |
|:-----------|:--------|:--------|
| [Streamlit](https://streamlit.io) | Web framework & UI | ≥ 1.38.0 |
| [Scikit-learn](https://scikit-learn.org) | ML algorithms, preprocessing, metrics | ≥ 1.3.0 |
| [Plotly](https://plotly.com/python/) | Interactive charts & visualizations | ≥ 5.15.0 |
| [Pandas](https://pandas.pydata.org) | Data manipulation & analysis | ≥ 2.0.0 |
| [NumPy](https://numpy.org) | Numerical computing | ≥ 1.24.0 |
| [SHAP](https://shap.readthedocs.io) | Model explainability | ≥ 0.42.0 |
| [XGBoost](https://xgboost.readthedocs.io) | Gradient boosting (classification & regression) | ≥ 2.0.0 |
| [LightGBM](https://lightgbm.readthedocs.io) | Fast gradient boosting | ≥ 4.0.0 |
| [UMAP](https://umap-learn.readthedocs.io) | Dimensionality reduction | ≥ 0.5.0 |
| [Matplotlib](https://matplotlib.org) | Static plotting backend | ≥ 3.7.0 |
| [Seaborn](https://seaborn.pydata.org) | Statistical visualization | ≥ 0.12.0 |

---

## 👨‍💻 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/pinshucodes">
        <img src="https://github.com/pinshucodes.png" width="120px;" alt="Priyanshu Shekhar"/><br />
        <sub><b>Priyanshu Shekhar</b></sub>
      </a><br />
      <a href="https://github.com/pinshucodes" title="GitHub">💻</a>
    </td>
    <td align="center">
      <a href="https://github.com/suppcodes">
        <img src="https://github.com/suppcodes.png" width="120px;" alt="Supriya Kumari"/><br />
        <sub><b>Supriya Kumari</b></sub>
      </a><br />
      <a href="https://github.com/suppcodes" title="GitHub">💻</a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Priyanshu Shekhar & Supriya Kumari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## ⭐ Show Your Support

If you found **ML Visualizer Lab** helpful or learned something new, please consider giving it a ⭐ on GitHub!

<p align="center">
  <a href="https://github.com/pinshucodes/MLVisualizerLab/stargazers">
    <img src="https://img.shields.io/github/stars/pinshucodes/MLVisualizerLab?style=social" alt="GitHub Stars"/>
  </a>
</p>

Your star helps others discover this project and motivates us to keep improving it. 🙌

> **Built with ❤️ using [Streamlit](https://streamlit.io)**

---

<p align="center">
  <sub>Made for learners, by learners. Happy exploring! 🧪✨</sub>
</p>
