"""
**Ejercicio9 -A-
"""
import skmultilearn
from skmultilearn.dataset import load_dataset
# here are the [ 'bibtex', 'enron',  'genbase', 'medical', 'yeast','rcv1subset4', 'delicious']
X_bibtex, y_bibtex, feature_names, label_names = load_dataset('bibtex', 'undivided')
X_enron, y_enron, feature_names, label_names = load_dataset('enron', 'undivided')
X_genbase, y_genbase, feature_names, label_names = load_dataset('genbase', 'undivided')
X_yeast, y_yeast, feature_names, label_names = load_dataset('yeast', 'undivided')
X_Corel5k, y_Corel5k, feature_names, label_names = load_dataset('Corel5k', 'undivided')

import pandas as pd
X_bibtex = pd.DataFrame(X_bibtex.toarray())
y_bibtex = y_bibtex.toarray()
# 'enron'
X_enron = pd.DataFrame(X_enron.toarray())
y_enron = y_enron.toarray()
# 'genbase'
X_genbase = pd.DataFrame(X_genbase.toarray())
y_genbase = y_genbase.toarray()
# 'yeast'
X_yeast = pd.DataFrame(X_yeast.toarray())
y_yeast = y_yeast.toarray()
# 'Corel5k'
X_Corel5k = pd.DataFrame(X_Corel5k.toarray())
y_Corel5k = y_Corel5k.toarray()

"""RakelD in bibtex -base classifier effect"""
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.ensemble import RakelD
from sklearn.model_selection import KFold
"""

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize an array to store predictions
predictions_bibtex_RakelD_GNB = np.empty_like(y_bibtex, dtype=int)
predictions_bibtex_RakelD_DT = np.empty_like(y_bibtex, dtype=int)
predictions_bibtex_RakelD_KNN = np.empty_like(y_bibtex, dtype=int)
predictions_bibtex_RakelD_RF = np.empty_like(y_bibtex, dtype=int)
for train_index, test_index in kf.split(X_bibtex, y_bibtex):
    X_train, X_test = X_bibtex.iloc[train_index], X_bibtex.iloc[test_index]
    y_train, y_test =  y_bibtex[train_index], y_bibtex[test_index]

    # Initialize the classifier
    classifier1 = RakelD(
        base_classifier=GaussianNB(),
        base_classifier_require_dense=[True, True],
        labelset_size=4
    )
    
    # Train the classifier
    classifier1.fit(X_train, y_train)

    # Predict on the test set
    predictions_bibtex_GN = classifier1.predict(X_test).toarray()  # Convert predictions to array
    
    # Initialize the Decision Tree classifier
    classifier2 = RakelD(
        base_classifier=DecisionTreeClassifier(),
        base_classifier_require_dense=[True, True],
        labelset_size=4
    )
    
    # Train the classifier
    classifier2.fit(X_train, y_train)
    
    # Convert the sparse matrix to a dense NumPy array
    predictions_bibtex_DT = classifier2.predict(X_test).toarray()
    
    # Initialize the K-Nearest Neighbors (KNN) classifier
    classifier3 = RakelD(
        base_classifier=KNeighborsClassifier(),
        base_classifier_require_dense=[True, True],
        labelset_size=4
    )
    
    # Train the classifier
    classifier3.fit(X_train, y_train)
    
    # Convert the sparse matrix to a dense NumPy array
    predictions_bibtex_KNN = classifier3.predict(X_test).toarray()
        # Initialize the classifier
    classifier4 = RakelD(
            base_classifier=RandomForestClassifier(),
            base_classifier_require_dense=[True, True],
            labelset_size=4
        )

    # Train the classifier
    classifier4.fit(X_train, y_train)
        # Convert label sets to NumPy array before prediction
    y_test_array = y_test.toarray()
    # Predict on the test set
    predictions_bibtex_RF = classifier4.predict(X_test).toarray()  # Convert predictions to array

    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_bibtex.shape[1]):
        predictions_bibtex_RakelD_GNB[test_index, label_index] = predictions_bibtex_GN[:, label_index]
        predictions_bibtex_RakelD_DT[test_index, label_index] = predictions_bibtex_DT[:, label_index]
        predictions_bibtex_RakelD_KNN[test_index, label_index] = predictions_bibtex_KNN[:, label_index]
        predictions_bibtex_RakelD_RF[test_index, label_index] = predictions_bibtex_RF[:, label_index]"""

""" ChainClassifier - base classifier effect"""
# using classifier chains
import time
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Registra el tiempo de inicio
start_time = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize an array to store predictions try 4 base classifier
predictions_Corel5k_ClassifierChain_RF = np.empty_like(y_Corel5k, dtype=int)
predictions_Corel5k_ClassifierChain_GNB = np.empty_like(y_Corel5k, dtype=int)
predictions_Corel5k_ClassifierChain_DT = np.empty_like(y_Corel5k, dtype=int)
predictions_Corel5k_ClassifierChain_KNN = np.empty_like(y_Corel5k, dtype=int)
for train_index, test_index in kf.split(X_Corel5k,y_Corel5k):
    X_train, X_test = X_Corel5k.iloc[train_index], X_Corel5k.iloc[test_index]
    y_train, y_test =  y_Corel5k[train_index], y_Corel5k[test_index]
    # initialize classifier chains multi-label classifier
    classifier1 = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=42))
    # Train the classifier
    classifier1.fit(X_train, y_train)
    # Predict on the test set
    predictions_Corel5k_RF = classifier1.predict(X_test).toarray()  # Convert predictions to array
    # initialize classifier chains multi-label classifier GNB
    classifier2 = ClassifierChain(GaussianNB())
    # Train the classifier
    classifier2.fit(X_train, y_train)
    # Predict on the test set
    predictions_Corel5k_GNB = classifier2.predict(X_test).toarray()  # Convert predictions to array
    # Decision tree
    classifier3 = ClassifierChain(DecisionTreeClassifier())
    # Train the classifier
    classifier3.fit(X_train, y_train)
    # Predict on the test set
    predictions_Corel5k_DT = classifier3.predict(X_test).toarray()  # Convert predictions to array
    # KNN
    classifier4 = ClassifierChain(KNeighborsClassifier())
    # Train the classifier
    classifier4.fit(X_train, y_train)
    # Predict on the test set
    predictions_Corel5k_KNN = classifier4.predict(X_test).toarray()  # Convert predictions to array

    # Store the predictions for each label in the corresponding test indices
    for label_index in range(y_yeast.shape[1]):
        predictions_Corel5k_ClassifierChain_RF[test_index, label_index] = predictions_Corel5k_RF[:, label_index]
        predictions_Corel5k_ClassifierChain_GNB[test_index, label_index] = predictions_Corel5k_GNB[:, label_index]
        predictions_Corel5k_ClassifierChain_DT[test_index, label_index] = predictions_Corel5k_DT[:, label_index]
        predictions_Corel5k_ClassifierChain_KNN[test_index, label_index] = predictions_Corel5k_KNN[:, label_index]

""" PERFORMANCE EVALUATION AND STORING THE RESULTS"""

import pandas as pd
import sklearn.metrics as metrics

# Assuming you have multiple models and their predictions stored in a list or dictionary
models = ["ClassifierChain_RF", "ClassifierChain_GNB","ClassifierChain_DT","ClassifierChain_KNN"]  # Add your model names

prediction_dict = {"ClassifierChain_RF":predictions_Corel5k_ClassifierChain_RF,
                  "ClassifierChain_GNB": predictions_Corel5k_ClassifierChain_GNB,
                   "ClassifierChain_DT": predictions_Corel5k_ClassifierChain_DT,
                   "ClassifierChain_KNN": predictions_Corel5k_ClassifierChain_KNN}
# Create a DataFrame to store the metrics
metrics_Corel5k = pd.DataFrame(columns=[
    'Hamming Loss', 'Recall (micro)', 'Recall (macro)',
    'Accuracy', 'F1-score (micro)', 'F1-score (macro)',
    'Precision (micro)', 'Precision (macro)'
])

# Add metrics for each model
for model in models:
    hamming_loss = metrics.hamming_loss(y_Corel5k, prediction_dict[model])
    recall_micro = metrics.recall_score(y_Corel5k, prediction_dict[model], average='micro')
    recall_macro = metrics.recall_score(y_Corel5k, prediction_dict[model], average='macro')
    accuracy = metrics.accuracy_score(y_Corel5k, prediction_dict[model])
    f1_micro = metrics.f1_score(y_Corel5k, prediction_dict[model], average='micro')
    f1_macro = metrics.f1_score(y_Corel5k, prediction_dict[model], average='macro')
    precision_micro = metrics.precision_score(y_Corel5k, prediction_dict[model], average='micro', zero_division=1)
    precision_macro = metrics.precision_score(y_Corel5k, prediction_dict[model], average='macro', zero_division=1)

    metrics_Corel5k.loc[model] = [
        hamming_loss, recall_micro, recall_macro,
        accuracy, f1_micro, f1_macro,
        precision_micro, precision_macro
    ]

metrics_Corel5k.to_csv('Corel5k_9A_Chainclassifier.csv', index=False)
