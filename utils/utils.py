# utils.py
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def accuracy_score_func(preds, labels):
    correct = (preds == labels).sum().float()
    total = len(preds)
    accuracy = correct / total
    return accuracy.item()
