from sklearn.metrics import precision_recall_curve

def optimize_threshold(y_true, probs, min_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    return thresholds[(recall >= min_recall).argmax()]
