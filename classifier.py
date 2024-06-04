import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])

print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

training_data = X_train
validation_data = X_test
training_labels = y_train
validation_labels = y_test

print(len(training_data), len(training_labels))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(3)

classifier.fit(training_data, training_labels)

print(classifier.score(validation_data, validation_labels))

for k in range(1, 101):
  classifier = KNeighborsClassifier(k)
  classifier.fit(training_data, training_labels)
  print(k, classifier.score(validation_data, validation_labels))

import matplotlib.pyplot as plt

k_list = range(1, 101)
accuracies = []

for k in range(1, 101):
  classifier = KNeighborsClassifier(k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show();
