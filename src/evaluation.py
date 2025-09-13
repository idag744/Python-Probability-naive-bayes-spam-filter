# src/evaluation.py

def get_true_positives(Y_true, Y_pred):
    return sum(1 for t, p in zip(Y_true, Y_pred) if t == 1 and p == 1)

def get_true_negatives(Y_true, Y_pred):
    return sum(1 for t, p in zip(Y_true, Y_pred) if t == 0 and p == 0)

def get_false_positives(Y_true, Y_pred):
    return sum(1 for t, p in zip(Y_true, Y_pred) if t == 0 and p == 1)

def get_recall(Y_true, Y_pred):
    total_positives = sum(Y_true)
    true_positives = get_true_positives(Y_true, Y_pred)
    return true_positives / total_positives if total_positives else 0

def get_precision(Y_true, Y_pred):
    true_positives = get_true_positives(Y_true, Y_pred)
    false_positives = get_false_positives(Y_true, Y_pred)
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0

def get_accuracy(Y_true, Y_pred):
    true_positives = get_true_positives(Y_true, Y_pred)
    true_negatives = get_true_negatives(Y_true, Y_pred)
    return (true_positives + true_negatives) / len(Y_true) if Y_true else 0

