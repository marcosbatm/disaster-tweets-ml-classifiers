# Disaster Tweets Classification

## Project Context

I developed this project as a practical assignment for the **Data Science** course within the *Ingeniería en Informática* degree at *Universidad de Buenos Aires (UBA)*. This work was done during the second semester of 2025.
The objective was to apply key concepts from the course by participating in a Kaggle machine learning competition.

The project consists of four parts:

1. Data Analysis and Visualization
2. Machine Learning Baseline
3. Machine Learning Models: XGBoost and RandomForest
4. Extras

## Dataset

This project uses the dataset from the Kaggle competition **"Natural Language Processing with Disaster Tweets"**: [check it here.](https://www.kaggle.com/c/nlp-getting-started)

To run the notebooks, you must download the following files directly from the competition page:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Place these files in the root of the repository (or adjust the paths as needed). Also, you must create the empty `./data/processed/` folders to allow data persistence between the several notebooks.

## Project Structure

After that, you should have a project structure similar to the following:

```bash
disaster-tweets-classification/
│
├── notebooks/
│ ├── 01_visualizations.ipynb
│ └── ...
│
├── data/
│ └── processed/
│
├── .gitignore
├── LICENSE
├── README.md
├── sample_submission.csv
├── test.csv
├── train.csv
└── requirements.txt
```

## Installation

Clone the repository and create a virtual environment, assuming you're on a Linux-based OS:

```bash
cd disaster-tweets-classification

python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

After that, you're ready to run the notebooks (If this is the first time you run a notebook, you might need to set up `jupyter`).
