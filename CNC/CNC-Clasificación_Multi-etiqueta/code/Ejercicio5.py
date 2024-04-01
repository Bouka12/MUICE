
""" **Ejecricio 5**: Familiarízate con la documentación scikit-multilearn en: http://scikit.ml/ Prueba los métodos
ML disponibles pertenecientes a las dos categorías (transformación y adaptación) que hemos
visto en teoría
"""

# Ejercicio 3
from skmultilearn.dataset import load_dataset
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency

# Define functions for calculations
def number_of_instances(X):
    return X.shape[0]

def number_of_attributes(X):
    return X.shape[1]

def number_of_labels(y):
    return y.shape[1]

# Function to calculate cardinality
def calculate_cardinality(y):
    return (1 / y.shape[0]) * np.sum(np.abs(y))
import pandas as pd
# Function to calculate density
def calculate_density(y):
    return np.mean(y)

# Function to calculate diversity
import scipy.sparse as sp
def calculate_diversity(y):
    unique_label_sets = len({tuple(row) for row in y.toarray()})  # Convert lil_matrix to array for set operations
    possible_label_sets = 2 ** y.shape[1]
    print("unique_label_sets: ", unique_label_sets)
    diversity = unique_label_sets / possible_label_sets
    return diversity

# Function to calculate imbalance ratio for a label
def calculate_imbalance_ratio(y):
    n = y.shape[1]
    irpl = np.zeros(n)

    for i in range(n):
        irpl[i] = y[:, i].sum()

    avg_ir = max(irpl) / irpl
    return np.mean(avg_ir)
# Function to convert label sets to integers
def label_sets_to_integers(y):
    return y.dot(1 << np.arange(y.shape[-1] - 1, -1, -1))

# Function to calculate ratio of unconditionally dependent label pairs using chi-square test
def calculate_dependent_label_pairs(y):
    print("shape of y", y.shape)
    label_pairs = list(combinations(range(y.shape[1]), 2))
    dependent_pairs = 0

    for pair in label_pairs:
        contingency_table = np.zeros((2, 2), dtype=np.int64)

        label_1 = y[:, pair[0]].toarray().flatten()
        label_2 = y[:, pair[1]].toarray().flatten()

        contingency_table[0, 0] = np.sum((label_1 == 0) & (label_2 == 0))  # Both labels absent
        contingency_table[0, 1] = np.sum((label_1 == 0) & (label_2 == 1))  # Label 1 absent, Label 2 present
        contingency_table[1, 0] = np.sum((label_1 == 1) & (label_2 == 0))  # Label 1 present, Label 2 absent
        contingency_table[1, 1] = np.sum((label_1 == 1) & (label_2 == 1))  # Both labels present

        _, p, _, _ = chi2_contingency(contingency_table)

        if p < 0.01:  # Considering 99% confidence level
            dependent_pairs += 1

    total_pairs = len(label_pairs)
    return dependent_pairs / total_pairs
# List of available datasets
datasets_list = ['scene']

def carFunc(datasetslist):
    # Iterate through datasets and calculate measures
    for dataset_name in datasetslist:
    # Load the dataset
        X, y, _, _ = load_dataset(dataset_name, 'undivided')
    # Print the data types of X and y
        print(f"Data type of X: {type(X)}")
        print(f"Data type of y: {type(y)}")
    # Calculate measures
        n_instances = number_of_instances(X)
        n_attributes = number_of_attributes(X)
        n_labels = number_of_labels(y)
        car = calculate_cardinality(y)
        den = calculate_density(y)
        div = calculate_diversity(y)
        avgIR = calculate_imbalance_ratio(y)
        rDep = calculate_dependent_label_pairs(y)

     # Print results
        print(f"Dataset: {dataset_name}")
        print(f"Number of instances (n): {n_instances}")
        print(f"Number of attributes (f): {n_attributes}")
        print(f"Number of labels (l): {n_labels}")
        print(f"Cardinality (car): {car}")
        print(f"Density (den): {den}")
        print(f"Diversity (div): {div}")
        print(f"Average Imbalance Ratio per label (avgIR): {avgIR}")
        print(f"Ratio of unconditionally dependent label pairs (rDep): {rDep}")
        print("\n")  # Add a separator between datasets


carFunc(datasets_list)

""" Proba los metodos del paquete scikit-multilearn"""

# scene
X, y, feature_names, label_names = load_dataset('scene', 'undivided')
import pandas as pd
X = pd.DataFrame(X.toarray())
y=y.toarray()
from sklearn.model_selection import train_test_split

# X and y are  features and labels
# test_size to the proportion of data you want to allocate to the test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
**Metodos de Transformacion**
from scipy.sparse import csr_matrix

# Convert training set to sparse matrices
X_train_sparse = csr_matrix(X_train)
y_train_sparse = csr_matrix(y_train)

# Convert testing set to sparse matrices
X_test_sparse = csr_matrix(X_test)
y_test_sparse = csr_matrix(y_test)
# BinaryRelevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

classifier = BinaryRelevance(
    classifier = SVC(),
    require_dense = [False, True]
)

# train
classifier.fit(X_train_sparse, y_train_sparse)

# predict
predictions_BinaryRelevance = classifier.predict(X_test_sparse)

# MultiOutputClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)).fit(X_train, y_train)
prediction_MultiOutputClassifier = clf.predict(X_test)

# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
# initialize label powerset multi-label classifier
classifier = LabelPowerset(RandomForestClassifier(n_estimators=100, random_state=42))
# train
classifier.fit(X_train, y_train)
# predict
predictions_LabelPowerset = classifier.predict(X_test)

# RakelD
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD

classifier1 = RakelD(
            base_classifier=GaussianNB(),
            base_classifier_require_dense=[True, True],
            labelset_size=4
        )

classifier1.fit(X_train, y_train)
predictions_RakelD = classifier1.predict(X_test)   

# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
# initialize classifier chains multi-label classifier
classifier2 = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=42))
# Training logistic regression model on train data
classifier2.fit(X_train, y_train)
# predict
predictions_ClassifierChain = classifier2.predict(X_test)

# Majority Voting Classifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB

classifier3 = MajorityVotingClassifier(
    clusterer = FixedLabelSpaceClusterer(clusters = [[1,2,3], [0, 2, 5], [4, 5]]),
    classifier = ClassifierChain(classifier=GaussianNB())
)
classifier3.fit(X_train,y_train)
predictions_MajorityVotingClassifier = classifier3.predict(X_test)"""

"""***Metodos de Apatacion*"""
from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=3)

        # train
classifier.fit(X_train, y_train)

        # predict
predictions_MLkNN = classifier.predict(X_test)
import sklearn.metrics as metrics
print("________________MLkNN Algorithm_________________")
print("Hamming Loss: ", metrics.hamming_loss(y_test, predictions_MLkNN))
"""f average='micro', it computes the recall globally by counting the total true positives, false negatives, and false positives.
If average='macro', it computes the recall for each label independently and then takes the unweighted average."""
print("Recall (micro): ", metrics.recall_score(y_test, predictions_MLkNN, average ='micro' ))
print("Recall (macro): ", metrics.recall_score(y_test, predictions_MLkNN, average ='macro' ))
print("Accuracy: ",metrics.accuracy_score(y_test, predictions_MLkNN))
print("F1-score (micro)", metrics.f1_score(y_test, predictions_MLkNN, average='micro'))
print("F1-score (macro)", metrics.f1_score(y_test, predictions_MLkNN, average='macro'))
print("Precision score (micro): ", metrics.precision_score(y_test, predictions_MLkNN, average='micro'))
print("Precision score (macro): ", metrics.precision_score(y_test, predictions_MLkNN, average='macro'))

""" **Try MLSVM** """
from skmultilearn.adapt import MLTSVM

classifier = MLTSVM(c_k = 2**-1)

        # train
classifier.fit(np.array(X_train) ,np.array( y_train))

        # predict
predictions_MLTSVM = classifier.predict(np.array(X_test))
print("________________MLTSVM Algorithm_________________")
print("Hamming Loss: ", metrics.hamming_loss(y_test, predictions_MLTSVM))
"""f average='micro', it computes the recall globally by counting the total true positives, false negatives, and false positives.
If average='macro', it computes the recall for each label independently and then takes the unweighted average."""
print("Recall (micro): ", metrics.recall_score(y_test, predictions_MLTSVM, average ='micro' ))
print("Recall (macro): ", metrics.recall_score(y_test, predictions_MLTSVM, average ='macro' ))
print("Accuracy: ",metrics.accuracy_score(y_test, predictions_MLTSVM))
print("F1-score (micro)", metrics.f1_score(y_test, predictions_MLTSVM, average='micro'))
print("F1-score (macro)", metrics.f1_score(y_test, predictions_MLTSVM, average='macro'))
print("Precision score (micro): ", metrics.precision_score(y_test, predictions_MLTSVM, average='micro'))
print("Precision score (macro): ", metrics.precision_score(y_test, predictions_MLTSVM, average='macro'))
""" **Try BRkNN** """
from skmultilearn.adapt import BRkNNaClassifier

classifier = BRkNNaClassifier(k=3)

        # train
classifier.fit(X_train, y_train)

        # predict
predictions_BRkNN = classifier.predict(X_test)
print("________________BRkNN Algorithm_________________")
print("Hamming Loss: ", metrics.hamming_loss(y_test, predictions_BRkNN))
"""f average='micro', it computes the recall globally by counting the total true positives, false negatives, and false positives.
If average='macro', it computes the recall for each label independently and then takes the unweighted average."""
print("Recall (micro): ", metrics.recall_score(y_test, predictions_BRkNN, average ='micro' ))
print("Recall (macro): ", metrics.recall_score(y_test, predictions_BRkNN, average ='macro' ))
print("Accuracy: ",metrics.accuracy_score(y_test, predictions_BRkNN))
print("F1-score (micro)", metrics.f1_score(y_test, predictions_BRkNN, average='micro'))
print("F1-score (macro)", metrics.f1_score(y_test, predictions_BRkNN, average='macro'))
print("Precision score (micro): ", metrics.precision_score(y_test, predictions_BRkNN, average='micro'))
print("Precision score (macro): ", metrics.precision_score(y_test, predictions_BRkNN, average='macro'))

import pandas as pd
import sklearn.metrics as metrics

# Assuming you have multiple models and their predictions stored in a list or dictionary
models = [ "Model7_Adapt (MLkNN)","Model8_Adapt (MLTSVM )","Model9_Adapt (BRkNN)"]  # Add your model names

prediction_dict = {
                   "Model7_Adapt (MLkNN)":predictions_MLkNN,
                   "Model8_Adapt (MLTSVM )":predictions_MLTSVM,
                    "Model9_Adapt (BRkNN)": predictions_BRkNN}
# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame(columns=[
    'Hamming Loss', 'Recall (micro)', 'Recall (macro)',
    'Accuracy', 'F1-score (micro)', 'F1-score (macro)',
    'Precision (micro)', 'Precision (macro)'
])

# Add metrics for each model
for model in models:
    hamming_loss = metrics.hamming_loss(y_test, prediction_dict[model])
    recall_micro = metrics.recall_score(y_test, prediction_dict[model], average='micro')
    recall_macro = metrics.recall_score(y_test, prediction_dict[model], average='macro')
    accuracy = metrics.accuracy_score(y_test, prediction_dict[model])
    f1_micro = metrics.f1_score(y_test, prediction_dict[model], average='micro')
    f1_macro = metrics.f1_score(y_test, prediction_dict[model], average='macro')
    precision_micro = metrics.precision_score(y_test, prediction_dict[model], average='micro')
    precision_macro = metrics.precision_score(y_test, prediction_dict[model], average='macro')

    metrics_df.loc[model] = [
        hamming_loss, recall_micro, recall_macro,
        accuracy, f1_micro, f1_macro,
        precision_micro, precision_macro
    ]

#metrics_df.to_csv('metrics_data_Adapt.csv', index=True)
