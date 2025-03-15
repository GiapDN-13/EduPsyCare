# EduPsyCare
Predicting Student Depression – A data science project aiming to detect and monitor depression risk in students, using Python, scikit-learn, Plotly, and Dash. Includes data analysis, model training, and an interactive dashboard for early intervention.

# Student Depression Risk Prediction and Monitoring Project

This project aims to **early detect** and **monitor depression risk** in students through data analysis and machine learning (AI-DS).

We use the **Student Depression Dataset** along with a standard process for data processing, model training, evaluation, and internal dashboard deployment.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Directory Structure](#directory-structure)
3. [How to Install & Set Up Environment](#how-to-install--set-up-environment)
4. [Project Pipeline](#project-pipeline-process)
5. [Running Guide](#running-guide)
6. [Results & Dashboard](#results--dashboard)
7. [Further Development](#further-development-guide)
8. [Contribute](#contribute)
9. [License](#license)

---

## Introduction
This project addresses the problem of **depression** in students – an important issue for universities and educational institutions. Objectives:
- **Analyze data** related to academic pressure, finances, lifestyle habits, etc.
- **Build a model** to predict depression (0/1) with algorithms such as Logistic Regression, Random Forest, etc.
- **Deploy a dashboard** so that the student affairs department can monitor and intervene early.

**Demo**:
(Path or screenshot of dashboard if deployed.)

---

## Folder Structure
```bash
depression_project/
├─ data/
│ └─ Student_Depression_Dataset.csv # Original data
├─ notebooks/
│ ├─ 01_EDA.ipynb # Data exploration notebook
│ └─ 02_Model_Training.ipynb
├─ models/
│ └─ depression_model.pkl # Model after training
├─ app/
│ └─ app.py # Dashboard code (Streamlit)
├─ utils/
│ └─ preprocessing.py # Data processing functions, utilities
├─ requirements.txt
└─ README.md # Project description file
