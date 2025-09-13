# src/main.py
import pandas as pd
from src.preprocessing import preprocess_text
from src.model import get_word_frequency, naive_bayes, log_naive_bayes
from src.evaluation import get_accuracy, get_precision, get_recall

# ---------------- Load Dataset ----------------
# Example CSV with 'text' and 'label' columns
df = pd.read_csv("data/emails.csv")  
X = df['text'].apply(preprocess_text).tolist()
Y = df['label'].tolist()  # 1 = spam, 0 = ham

# ---------------- Split Dataset ----------------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------------- Compute Word Frequencies ----------------
word_frequency = get_word_frequency(X_train, Y_train)
class_frequency = {'ham': sum(y == 0 for y in Y_train), 'spam': sum(y == 1 for y in Y_train)}

# ---------------- Evaluate Models ----------------
# Standard Naive Bayes
Y_pred_nb = [naive_bayes(email, word_frequency, class_frequency) for email in X_test]
# Log Naive Bayes
Y_pred_log_nb = [log_naive_bayes(email, word_frequency, class_frequency) for email in X_test]

# Accuracy
accuracy_nb = get_accuracy(Y_test, Y_pred_nb)
accuracy_log_nb = get_accuracy(Y_test, Y_pred_log_nb)

# Precision
precision_nb = get_precision(Y_test, Y_pred_nb)
precision_log_nb = get_precision(Y_test, Y_pred_log_nb)

# Recall
recall_nb = get_recall(Y_test, Y_pred_nb)
recall_log_nb = get_recall(Y_test, Y_pred_log_nb)

print(f"Standard Naive Bayes -> Accuracy: {accuracy_nb:.4f}, Precision: {precision_nb:.4f}, Recall: {recall_nb:.4f}")
print(f"Log Naive Bayes -> Accuracy: {accuracy_log_nb:.4f}, Precision: {precision_log_nb:.4f}, Recall: {recall_log_nb:.4f}")

