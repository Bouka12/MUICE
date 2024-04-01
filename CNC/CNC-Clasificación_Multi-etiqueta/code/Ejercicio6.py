"""
When working with Machine Learning, cross-validation is a crucial step to assess the model's performance.
 There are several functions in scikit-learn that facilitate this process:
"""
# Kfold
"""
KFold:
KFold is a class in scikit-learn that splits the dataset into k folds for cross-validation. Each fold is used as a test set 
exactly once. Below is an example:
"""

# Ejercicio 6
from skmultilearn.dataset import load_dataset
from itertools import combinations
import numpy as np
# scene
X, y, feature_names, label_names = load_dataset('scene', 'undivided')
import pandas as pd
X = pd.DataFrame(X.toarray())
y=y.toarray()
from sklearn.model_selection import train_test_split

# X and y are  features and labels
# test_size to the proportion of data you want to allocate to the test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import KFold
from skmultilearn.adapt import BRkNNaClassifier
import sklearn.metrics as metrics

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize an array to store predictions
all_predictions_BRkNN = np.empty_like(y, dtype=int)

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test =  y[train_index], y[test_index]

    # Initialize the classifier
    classifier = BRkNNaClassifier(k=3)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict on the test set
    predictions_BRkNN = classifier.predict(X_test).toarray()  # Convert predictions to array

    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y.shape[1]):
        all_predictions_BRkNN[test_index, label_index] = predictions_BRkNN[:, label_index]

print("Hamming Loss: ", metrics.hamming_loss(y, all_predictions_BRkNN))

#f average='micro', it computes the recall globally by counting the total true positives, false negatives, and false positives.
#If average='macro', it computes the recall for each label independently and then takes the unweighted average.

print("Recall (micro): ", metrics.recall_score(y, all_predictions_BRkNN, average ='micro' ))
print("Recall (macro): ", metrics.recall_score(y, all_predictions_BRkNN, average ='macro' ))
print("Accuracy: ",metrics.accuracy_score(y, all_predictions_BRkNN))
print("F1-score (micro)", metrics.f1_score(y, all_predictions_BRkNN, average='micro'))
print("F1-score (macro)", metrics.f1_score(y, all_predictions_BRkNN, average='macro'))
print("Precision score (micro): ", metrics.precision_score(y, all_predictions_BRkNN, average='micro'))
print("Precision score (macro): ", metrics.precision_score(y, all_predictions_BRkNN, average='macro'))




# cross_validate:
# cross_validate is a function that performs cross-validation and returns evaluation metrics. You can specify multiple metrics and 
# get results for each fold. Here's an example:

"""
from sklearn.model_selection import cross_validate
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
# Define a custom scorer for micro and macro averaging
def custom_scorer(estimator, X, y):
    # Predict on the input data
    y_pred = estimator.predict(X)
    
    # Ensure that both y_true and y_pred are numpy arrays
    y_true = np.array(y)

    # Compute precision, recall, and f1-score for each label
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=1)

    # Compute micro-averaged precision, recall, and f1-score
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro',zero_division=1)

    # Compute macro-averaged precision, recall, and f1-score
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)

    # Compute Hamming loss
    hamming_loss_value = hamming_loss(y_true, y_pred)

    return {
        'precision_micro': micro_precision,
        'recall_micro': micro_recall,
        'f1_micro': micro_f1,
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
        'f1_macro': macro_f1,
        'hamming_loss': hamming_loss_value
    }

# Initialize the classifier
classifier = BRkNNaClassifier(k=3)

# Specify the custom scorer in the cross_validate function
scoring = custom_scorer
cv_results = cross_validate(classifier, X, y, scoring=scoring, cv=5)

# Print the results
for metric, values in cv_results.items():
    print(f"{metric}: {np.mean(values)}")"""



# the above is working done


# cross_val_score:
# cross_val_score simplifies cross-validation and returns an array of results for a single metric. Here's a simple example:"""
# cross_val_score
"""
from sklearn.model_selection import cross_val_score
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
from sklearn.metrics import make_scorer, accuracy_score
def custom_scorer(estimator, X, y):
    # Predict on the input data
    y_pred = estimator.predict(X)
    
    # Ensure that both y_true and y_pred are numpy arrays
    y_true = np.array(y)

    # Compute precision, recall, and f1-score for each label
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=1)

    # Compute micro-averaged precision, recall, and f1-score
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro',zero_division=1)

    # Compute macro-averaged precision, recall, and f1-score
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)

    # Compute Hamming loss
    hamming_loss_value = hamming_loss(y_true, y_pred)

    return  micro_precision
# Now, use BRkNNaWrapper in cross_val_score
classifier = BRkNNaClassifier(k=3)
# You may want to handle cases where 'classes_' is None, for example:
scoring = custom_scorer
cv_scores = cross_val_score(classifier, X, y, scoring=scoring, cv=5)
print(f"Precision (micro): {cv_scores.mean()}")"""
#_____________________________________________________________________________________________________________#

# make_scorer:
# make_scorer converts custom loss or metric functions into scikit-learn compatible functions. This is useful when using your 
# own evaluation metric:
# make_scorer
"""
from sklearn.model_selection import GridSearchCV, cross_val_score
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, recall_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score

from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
# 1. Example of using make_scorer in cross_val_score method

def custom_f1_macro(y_true, y_pred, **kwargs):
    return f1_score(y_true, y_pred, average='macro', **kwargs)

# Use MLkNN in cross_val_score
classifier = MLkNN()
scorer = make_scorer(custom_f1_macro, greater_is_better=True)
cv_scores = cross_val_score(classifier, X, y, scoring=scorer, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score_f1-score (macro):", cv_scores.mean())


# 2. Example of using make_scorer in GreadSearch

def custom_accuracy_score(y_true, y_pred, **kwargs):
    return accuracy_score(y_true, y_pred, **kwargs)

# Use MLkNN in GridSearchCV with custom scoring
parameters = {'k': range(1, 3), 's': [0.5, 0.7, 1.0]}
scorer = make_scorer(custom_accuracy_score, greater_is_better=True)
clf = GridSearchCV(MLkNN(), parameters, scoring=scorer)
clf.fit(X, y)

print("GridSearchCV Best Parameters:", clf.best_params_)
print("GridSearchCV Best Score_accuracy:", clf.best_score_)"""
