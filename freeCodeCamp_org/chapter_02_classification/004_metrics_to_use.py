''''''

'''
Metrics:
    https://github.com/mrdbourke/pytorch-deep-learning/blob/main/02_pytorch_classification.ipynb

Accuracy
    Out of 100 predictions, how many does your model get correct? E.g. 95% accuracy means it gets 95/100 predictions correct.
    torchmetrics.Accuracy() or sklearn.metrics.accuracy_score()

Precision
    Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0).
    torchmetrics.Precision() or sklearn.metrics.precision_score()

Recall
    Proportion of true positives over total number of true positives and false negatives (model predicts 0 when it should've been 1). Higher recall leads to less false negatives.
    torchmetrics.Recall() or sklearn.metrics.recall_score()

F1-score
    Combines precision and recall into one metric. 1 is best, 0 is worst.
    torchmetrics.F1Score() or sklearn.metrics.f1_score()

Confusion matrix
    Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).
    torchmetrics.ConfusionMatrix or sklearn.metrics.plot_confusion_matrix()

Classification report
    Collection of some of the main classification metrics such as precision, recall and f1-score.
    sklearn.metrics.classification_report()
'''

'''
Beyond Accuracy: Precision And Recall
https://willkoehrsen.github.io/statistics/learning/beyond-accuracy-precision-and-recall/
https://medium.com/@priyankads/beyond-accuracy-recall-precision-f1-score-roc-auc-6ef2ce097966
'''

'''
precision recall tradeoff
'''
