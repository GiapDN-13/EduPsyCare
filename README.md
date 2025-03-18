# EduPsyCare: Student Depression Prediction and Analysis

## Project Description

EduPsyCare is a data science and machine learning project designed to predict depression risk among students based on various features such as CGPA, work/study hours, financial stress, work pressure, sleep duration, and other psychological factors.

This interactive dashboard aims to help school departments monitor, supervise, and identify students at risk of depression for timely support and intervention.

## Bookmarks

- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [How to Use the Dashboard](#how-to-use-the-dashboard)
- [Version Control with Git](#version-control-with-git)
- [Managing Python Libraries](#managing-python-libraries)
- [Contact](#contact)
- [Contribution](#contribution)

## Project Structure

```
EduPsyCare/
├── app/
│   └── app.py                               # Streamlit Dashboard
├── data/
│   ├── Student_Depression_Dataset.csv       # Raw Data
│   └── Student_Depression_Dataset_clean.csv # Cleaned Data after preprocessing
├── models/
│   └── depression_model.pkl                 # Best trained model
├── notebooks/
│   ├── 00_check_env.ipynb                   # Environment Check Notebook
│   ├── EduPsyCare_Data_Analysis.ipynb       # EDA and Data Analysis Notebook
│   └── Model_Training_and_Evaluation.ipynb  # Notebook for model training and evaluation
├── requirements.txt                         # Python libraries list
├── .gitignore                               # Ignore unnecessary files
└── README.md                                # This document
```

## Installation and Usage

### System Requirements

- **Python:** >= 3.11
- **pip:** Python package manager
- **Git:** Version control tool

### Installation

```bash
git clone https://github.com/YourUsername/EduPsyCare.git
cd EduPsyCare
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app/app.py
```

## How to Use the Dashboard

The dashboard includes 3 main tabs:

### Prediction
- Enter student details for depression risk prediction.
- Displays "Pressure Index" and "Prediction Probability" in gauge charts.

### EDA & Feature Relationships
- Visualizes data distributions using histogram, boxplot, and scatter matrix.
- Interactive correlation heatmap for deeper insights.

### Model Evaluation
- Radar Chart, Confusion Matrix, ROC Curve, and Precision-Recall Curve for performance analysis.

## Version Control with Git

### Commit Changes
```bash
git add .
git commit -m "Update with latest improvements"
```

### Tagging Versions
```bash
git tag -a v1.0.a -m "Version 1.0.a - Final Dashboard Improvements"
git push origin main --tags
```

### Deleting Tags (If Needed)
```bash
git tag -d v1.1.0
git push origin :refs/tags/v1.1.0
```

## Managing Python Libraries

To update the `requirements.txt` file with all necessary libraries:

```bash
pip install pipreqs nbconvert
jupyter nbconvert --to script notebooks/Model_Training_and_Evaluation.ipynb
pipreqs . --force
pip install -r requirements.txt
```

## Contact

- **Name:** Dang Nguyen Giap
- **Email:** dangnguyengiap2004@gmail.com
- **GitHub:** [GiapDN-13](https://github.com/GiapDN-13)

## Contribution

Contributions are welcome. Feel free to open a pull request or issue to discuss improvements.

Thank you for using EduPsyCare!

