def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            TP += 1
        elif prediction[i] and not ground_truth[i]:
            FP += 1
        elif not prediction[i] and ground_truth[i]:
            FN += 1
        elif not prediction[i] and not ground_truth[i]:
            TN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    total = len(prediction)
    correct = 0
    for i in range(total):
        if prediction[i] == ground_truth[i]:
            correct += 1
        
    return correct / total
