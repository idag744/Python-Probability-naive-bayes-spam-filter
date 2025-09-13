# src/model.py
import numpy as np

def get_word_frequency(X, Y):
    """
    Calculate frequency of each word in spam (1) and ham (0) emails.
    """
    word_dict = {}
    for email, cls in zip(X, Y):
        email = set(email)  # remove duplicates
        for word in email:
            if word not in word_dict:
                word_dict[word] = {"spam": 0, "ham": 0}
            if cls == 0:
                word_dict[word]["ham"] += 1
            else:
                word_dict[word]["spam"] += 1
    return word_dict

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    Compute P(word|class)
    """
    return word_frequency[word][cls] / class_frequency[cls]

def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Compute P(email|class) using naive assumption
    """
    prob = 1
    for word in treated_email:
        if word in word_frequency:
            prob *= prob_word_given_class(word, cls, word_frequency, class_frequency)
    return prob

def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood=False):
    """
    Standard Naive Bayes classifier
    """
    prob_email_given_spam = prob_email_given_class(treated_email, "spam", word_frequency, class_frequency)
    prob_email_given_ham = prob_email_given_class(treated_email, "ham", word_frequency, class_frequency)
    
    total_emails = class_frequency["spam"] + class_frequency["ham"]
    p_spam = class_frequency["spam"] / total_emails
    p_ham = class_frequency["ham"] / total_emails
    
    spam_likelihood = p_spam * prob_email_given_spam
    ham_likelihood = p_ham * prob_email_given_ham
    
    if return_likelihood:
        return (spam_likelihood, ham_likelihood)
    return 1 if spam_likelihood >= ham_likelihood else 0

# ---------------- LOG VERSION ----------------
def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Compute log(P(email|class)) for numerical stability
    """
    log_prob = 0
    for word in treated_email:
        if word in word_frequency:
            log_prob += np.log(prob_word_given_class(word, cls, word_frequency, class_frequency))
    return log_prob

def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood=False):
    """
    Log version of Naive Bayes
    """
    log_prob_email_given_spam = log_prob_email_given_class(treated_email, "spam", word_frequency, class_frequency)
    log_prob_email_given_ham = log_prob_email_given_class(treated_email, "ham", word_frequency, class_frequency)
    
    total_emails = class_frequency["spam"] + class_frequency["ham"]
    p_spam = class_frequency["spam"] / total_emails
    p_ham = class_frequency["ham"] / total_emails
    
    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam
    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham
    
    if return_likelihood:
        return (log_spam_likelihood, log_ham_likelihood)
    return 1 if log_spam_likelihood >= log_ham_likelihood else 0

