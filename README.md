# Naive Bayes Spam Filter

A from-scratch implementation of the **Naive Bayes classifier** for spam detection.  
This project demonstrates fundamental **Natural Language Processing (NLP)** and **Machine Learning** concepts using Python, without relying on external ML libraries like scikit-learn.  

The goal is to show:
- Data preprocessing and text cleaning
- Implementing Naive Bayes manually
- Handling numerical underflow with log probabilities
- Evaluating models with accuracy, precision, and recall
- Structuring code for readability and reusability

---

## 📂 Project Structure

naive-bayes-spam-filter/
│── data/
│   └── emails.csv              # dataset (not included in repo, but README explains how to get it)
│
│── src/
│   ├── preprocessing.py        # preprocessing steps
│   ├── model.py                # naive bayes model implementation
│   ├── evaluation.py           # evaluation metrics (accuracy, precision, recall)
│   └── main.py                 # entry point to train/test the model
│
│── notebooks/
│   └── exploration.ipynb       # optional: dataset exploration notebook
│
│── tests/
│   └── test_model.py           # simple unit tests
│
│── requirements.txt            # dependencies
│── README.md                   # project overview
│── .gitignore                  # ignore cache, pyc files, datasets


---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/naive-bayes-spam-filter.git
cd naive-bayes-spam-filter

### 2. Install dependencies
pip install -r requirements.txt


### 3. Run the model
python src/main.py

## 📊 Example Output
Accuracy: 0.987
Precision: 0.981
Recall: 0.972

## 🛠️ Technologies Used
Python 3
NumPy
Pandas
NLTK (tokenization & stopword removal)
