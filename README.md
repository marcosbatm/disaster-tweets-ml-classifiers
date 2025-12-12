# Disaster Tweets Classification

## Project Context

I developed this project as a practical assignment for the **Data Science** course within the *Ingeniería en Informática* degree at *Universidad de Buenos Aires (UBA)*. This work was done during the second semester of 2025.
The objective was to apply key concepts from the course by participating in a Kaggle machine learning competition.

The project consists of four parts:

1. Data Analysis and Visualization
2. Machine Learning Baseline: LogisticRegression
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
├── data/
│ └── processed/
├── docs/
│ ├── docs.md
│ └── docs.pdf
├── notebooks/
│ ├── 01_visualizations.ipynb
│ └── ...
├── visualizations/
│ ├── 01_wordCloud.png
│ └── ...
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── sample_submission.csv
├── test.csv
└── train.csv
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

After that, you're ready to run the notebooks (If this is the first time you run a notebook, you might need to set up `jupyter`). **Keep in mind some of tihs notebooks have cells that may take several minutes up to one or two hours.**

## Notebooks

All the notebooks are self-explanatory, containing text-cells with descriptions of the objective of the notebook and also sharing the thought process of each code step. However, they are all written in Spanish. A thorough explanation in English is available in the [documentation](docs/docs.pdf).

## License

This repository is licensed under the GNU GPL v3 [license](LICENSE).
