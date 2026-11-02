# SCAM SLEUTH

**Intelligent Fake Job and Internship Detection Using Data Mining**

SCAM SLEUTH explores job-posting data to identify patterns associated with postings labelled as fraudulent in a research dataset. It uses text preprocessing, TF-IDF, Logistic Regression, K-Means clustering, visualisation and optional Apriori association-rule mining.

## Repository contents
- `SCAM_SLEUTH.ipynb` – reproducible analysis notebook
- `requirements.txt` – Python dependencies
- `graphs/` – report figures and project diagrams
- `SCAM_SLEUTH_Final_Student_Style_Report.pdf` – final academic report (upload separately from the notebook package if desired)
- `SCAM_SLEUTH_Final_Student_Style_Report.docx` – editable report (upload separately)

## Dataset
The notebook expects `fake_job_postings.csv` or `DataSet.csv` in the same directory. The dataset is not included in this repository. Obtain it from its original source and review its licence/terms before use or redistribution. The supplied copy contains 17,880 rows and the target column `fraudulent` (`f` = legitimate, `t` = labelled fraudulent).

## Run locally or in Google Colab
1. Download/obtain the dataset and place it beside the notebook.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `SCAM_SLEUTH.ipynb` in Jupyter or upload it to Google Colab.
4. Run cells from top to bottom.
5. Generated charts and `model_summary.csv` are saved under `outputs/`.

## Important limitation
A model prediction is only a screening signal, not proof of fraud. Dataset labels may be imperfect or outdated. Independently verify employers and never pay recruitment fees or share sensitive information based only on a model result.
