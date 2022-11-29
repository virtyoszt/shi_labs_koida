import numpy as np
from sklearn import preprocessing
Input_labels = ['purple', 'red', 'yellow', 'blue', 'black', 'yellow', 'blue', 'white', 'red']
encoder = preprocessing.LabelEncoder()
encoder.fit(Input_labels)
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)
test_labels = ['blue', 'yellow', 'red']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [1, 3, 2, 5]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))
