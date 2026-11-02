# SCAM SLEUTH
### Intelligent Fake Job and Internship Detection Using Data Mining

SCAM SLEUTH is an academic data-mining project that explores how text analysis and machine-learning techniques can help screen online job and internship advertisements for patterns associated with fraudulent postings.

> **Important:** A model prediction is only a screening signal. It does not prove that an employer or job advertisement is fraudulent. Always verify an organisation independently before sharing personal information, paying fees, or accepting an offer.

## Project objectives

- Explore and understand a labelled job-posting dataset.
- Inspect missing values and the distribution of the target labels.
- Clean and combine relevant job-advertisement text fields.
- Build a baseline text classifier using TF-IDF and Logistic Regression.
- Evaluate the classifier with precision, recall, F1-score, accuracy, and a confusion matrix.
- Explore text groupings with K-Means clustering.
- Examine frequent terms in postings carrying the fraudulent label.
- Optionally explore association rules from selected categorical attributes.

## Dataset

The project uses a version of the **Fake Job Postings / EMSCAD** dataset supplied as a CSV file. The notebook expects the target column `fraudulent` and supports either `fake_job_postings.csv` or `DataSet.csv` as the local filename.

The dataset is not included in this repository. Please obtain it from its original source and review its licence and terms before using or redistributing it. The notebook checks for expected text fields and reports an error if required columns are missing.

## Methods and tools

| Area | Method / tool |
|---|---|
| Data handling | Python, pandas, NumPy |
| Text cleaning | HTML removal, URL normalisation, whitespace and character cleaning |
| Text representation | TF-IDF with unigrams and bigrams |
| Classification | Logistic Regression with class weighting |
| Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |
| Unsupervised exploration | K-Means clustering and Truncated SVD projection |
| Visualisation | Matplotlib |
| Optional pattern mining | Apriori association rules with `mlxtend` |

The notebook places TF-IDF vectorisation inside a scikit-learn pipeline, so the vocabulary is learned from the training split rather than the held-out test data. It uses a stratified 80/20 train-test split with a fixed random seed for reproducibility.

## Repository contents

- `SCAM_SLEUTH.ipynb` — step-by-step notebook for loading, exploring, modelling, and evaluating the dataset.
- `requirements.txt` — Python packages required for the main notebook workflow.
- `SCAM_SLEUTH_Final_Student_Style_Report.pdf` — project report.
- `SCAM SLEUTH Student Style Report.docx` — editable report.
- `scam_graph_01_label_distribution.png` — labelled class distribution.
- `scam_graph_02_missing_values.png` — missing-value overview.
- `scam_graph_03_fraud_terms.png` — frequent terms in labelled fraudulent postings.
- `scam_graph_04_confusion_matrix.png` — classifier confusion matrix.
- `scam_graph_05_metrics.png` — model evaluation metrics.
- `scam_sleuth_architecture.png`, `scam_sleuth_features.png`, `scam_sleuth_workflow.png` — project diagrams.

## Run the notebook

### Option 1: Google Colab

1. Download `SCAM_SLEUTH.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Choose **File → Upload notebook** and select the downloaded notebook.
4. Upload the dataset CSV to the Colab session, or update `DATA_PATH` in the notebook to the dataset location.
5. Run the cells from top to bottom.

### Option 2: Run locally

1. Install Python 3.10 or later.
2. Download or clone this repository.
3. Place the dataset CSV beside the notebook.
4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Start Jupyter:

```bash
jupyter notebook
```

6. Open `SCAM_SLEUTH.ipynb` and run the cells in order.

The notebook saves generated charts and a compact model summary in an `outputs/` folder. Apriori analysis is optional and requires `mlxtend`; install it in the notebook if you want to run that section.

## Understanding the results

Do not judge an imbalanced classification task by accuracy alone. Review:

- **Precision:** Of the postings flagged by the model, how many carry the fraudulent label in the test data?
- **Recall:** Of the postings carrying the fraudulent label, how many did the model identify?
- **F1-score:** A combined measure of precision and recall.
- **Confusion matrix:** Shows correct and incorrect predictions for both classes.

K-Means clusters represent similarities in text, not verified scam categories. Frequent words are descriptive patterns, not proof of wrongdoing. Association-rule results, if produced, describe co-occurrence patterns in the selected attributes and should not be treated as causal findings.

## Limitations and responsible use

- Dataset labels may be incomplete, outdated, or reflect the original dataset's collection and labelling process.
- Language and recruitment practices change; performance on this dataset may not transfer to current job platforms.
- False positives and false negatives are possible.
- The model does not verify a company's identity, registration, contact details, or intentions.
- Do not use predictions as the sole basis for accusing an employer or rejecting an opportunity.
- Do not upload personal information, credentials, or private recruitment correspondence to this public repository.

## Academic project

**Project title:** SCAM SLEUTH – Intelligent Fake Job and Internship Detection Using Data Mining  
**Student:** Gurnoor Kaur  
**Programme:** BCA  
**Institution:** Chandigarh University  
**Academic session:** 2026–2027

---

This repository is maintained for academic learning and demonstration. Contributions and suggestions for improving the analysis are welcome.
