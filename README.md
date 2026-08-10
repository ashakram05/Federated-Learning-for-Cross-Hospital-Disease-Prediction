# Federated Learning for Cross-Hospital Disease Prediction

> A research-based Final Year Project exploring privacy-preserving disease prediction using Federated Learning across distributed healthcare environments.

## 📌 Overview

Healthcare institutions generate large amounts of sensitive patient data that can be valuable for developing machine learning models. However, directly sharing patient data between hospitals raises significant privacy and security concerns.

This project investigates the use of **Federated Learning (FL)** to train disease prediction models across distributed healthcare clients without requiring their raw patient data to be centralized.

The project focuses on two major healthcare prediction tasks:

* 🫀 **Cardiovascular Disease Prediction**
* 🩺 **Diabetes Prediction**

The ultimate goal is to investigate whether federated models can achieve competitive predictive performance while keeping sensitive healthcare data distributed.

---

## 🎯 Research Problem

Traditional machine learning approaches typically require data from different healthcare institutions to be collected into a central location.

For healthcare data, this can introduce:

* Privacy concerns
* Data-sharing limitations
* Security risks
* Institutional data ownership issues
* Difficulty in combining data from different healthcare environments

This project explores **Federated Learning** as an alternative approach where participating clients can collaboratively train a global model while keeping their local data decentralized.

---

## 🔬 Research Objectives

The main objectives of this project are to:

1. Investigate Federated Learning for healthcare disease prediction.
2. Develop predictive models for cardiovascular disease and diabetes.
3. Preprocess and prepare healthcare datasets for machine learning.
4. Simulate distributed healthcare clients.
5. Investigate the impact of non-IID data distributions on federated learning.
6. Compare centralized and federated learning performance.
7. Investigate privacy-preserving techniques for federated healthcare learning.
8. Evaluate model performance using appropriate classification metrics.
9. Develop an interface for interacting with the final prediction system.

---

## 📊 Datasets

### 1. Cardiovascular Disease Dataset

The cardiovascular disease dataset contains **70,000 patient records** and multiple demographic, clinical, and lifestyle-related features.

**Source:** Kaggle

[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)

The dataset is used to investigate binary cardiovascular disease prediction.

### 2. Diabetes 130-US Hospitals Dataset

The Diabetes 130-US Hospitals dataset contains approximately **101,000 healthcare encounters** collected from **130 US hospitals** between 1999 and 2008.

**Source:** UCI Machine Learning Repository

[Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

The dataset provides a particularly relevant foundation for investigating distributed healthcare learning because it originates from multiple hospitals.

---

## 🧹 Current Progress

### Phase 1 — Data Preprocessing ✅

The initial data preprocessing stage has been completed.

Current work includes:

* Dataset loading and inspection
* Dataset structure analysis
* Data type identification
* Missing-value analysis
* Duplicate analysis
* Feature inspection
* Data cleaning
* Categorical feature handling
* Numerical feature preparation
* Target variable preparation
* Preparation of datasets for subsequent machine learning experiments

> The repository will be updated progressively as each research and development stage is completed.

---

## 🧠 Planned Methodology

The project will progressively investigate the following pipeline:

```text
Raw Healthcare Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Preparation
        │
        ▼
Centralized Baseline Models
        │
        ▼
Client / Hospital Simulation
        │
        ▼
Non-IID Data Distribution
        │
        ▼
Federated Learning
        │
        ▼
Global Model Aggregation
        │
        ▼
Performance Evaluation
        │
        ▼
Privacy & Security Experiments
        │
        ▼
Prediction Interface
```

---

## 🏥 Federated Learning Concept

The intended architecture simulates multiple healthcare institutions as independent clients.

```text
                  ┌─────────────────────┐
                  │   Central Server    │
                  │     FedAvg          │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         Hospital 1     Hospital 2     Hospital 3
          Client 1       Client 2       Client 3
              │              │              │
         Local Data     Local Data     Local Data
              │              │              │
         Local Model    Local Model    Local Model
              └──────────────┼──────────────┘
                             │
                             ▼
                       Global Model
```

The federated system will allow participating clients to train locally and contribute model updates to a global model without directly exchanging their raw training data.

---

## 🤖 Machine Learning

The project will investigate machine learning approaches suitable for binary disease prediction.

Planned experiments include:

* Logistic Regression
* Feedforward Neural Networks
* Centralized learning
* Federated learning
* Multiple simulated clients
* IID and non-IID data distributions

---

## 🔐 Privacy & Security

Because healthcare data is highly sensitive, privacy is an important component of this research.

Planned investigations include:

* Federated Learning privacy limitations
* Gradient leakage
* Membership inference attacks
* Differential privacy
* Privacy–utility trade-offs

These experiments will help evaluate whether federated learning alone is sufficient for protecting sensitive healthcare information.

---

## 📈 Evaluation

Model performance will be evaluated using appropriate classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Federated models will also be compared with centralized baseline models to investigate the performance impact of decentralized training.

---

## 🛠️ Technology Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* TensorFlow / Keras

### Federated Learning

* Flower

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Planned Application

* Streamlit / Flask / FastAPI

---

## 📁 Project Structure

The repository is being developed progressively alongside the research.

```text
federated-disease-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── data-preprocessing/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── federated/
│   ├── privacy/
│   └── evaluation/
│
├── experiments/
│
├── results/
│
├── docs/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚧 Project Status

**Current Stage: Data Preprocessing**

The project is currently in the initial research and data preparation phase.

Future stages will include:

* [ ] Exploratory Data Analysis
* [ ] Centralized baseline models
* [ ] Client/hospital simulation
* [ ] Non-IID data partitioning
* [ ] Federated Learning implementation
* [ ] Model aggregation
* [ ] Centralized vs Federated comparison
* [ ] Privacy experiments
* [ ] Security evaluation
* [ ] Final prediction interface
* [ ] Experimental analysis and documentation

---

## 📚 Research Direction

This project follows a research-and-development approach, with emphasis on understanding the practical challenges of applying Federated Learning to distributed healthcare data.

Particular attention will be given to:

**Data heterogeneity → Non-IID distributions → Federated optimization → Model performance → Privacy → Security**

---

## 👩‍💻 Project

**Final Year Project (FYP)**
**Federated Learning for Cross-Hospital Disease Prediction**

Developed as a research-based undergraduate project in Computer Science.

---

## 📜 License

This project is developed for academic and research purposes.
