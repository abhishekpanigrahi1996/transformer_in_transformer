import numpy as np

# For question answering
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_metric(predictions, metric_name):
    if metric_name == "accuracy":
        if isinstance(predictions[0].correct_candidate, list):
            return np.mean([pred.predicted_candidate in pred.correct_candidate for pred in predictions])
        else:
            return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])
    elif metric_name == "em":
        return np.mean([any([normalize_answer(ans) == normalize_answer(pred.predicted_candidate) for ans in pred.correct_candidate]) for pred in predictions])
    elif metric_name == "substring_em":
        return np.mean([any([normalize_answer(ans) in normalize_answer(pred.predicted_candidate) for ans in pred.correct_candidate]) for pred in predictions])
        
        