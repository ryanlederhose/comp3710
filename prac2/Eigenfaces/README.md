# Eigenfaces
In this task, we compute Eigenfaces - the PCA of human faces.

# Results
Eigenfaces (PyTorch implementation):

![eigenfaces-pytorch](https://github.com/ryanlederhose/comp3710/assets/112144274/8ae3aa36-713f-44fc-932b-2566402c65f0)

Compactness (PyTorch implementation):

![compactness-pytorch](https://github.com/ryanlederhose/comp3710/assets/112144274/b7143c2f-a50b-4181-b030-efd4396d619e)

## Training Results of Random Forest
```bash
Predictions [3 3 4 3 3 3 3 1 3 3 3 1 3 3 3 3 3 3 3 4 1 1 3 3 2 1 3 3 3 3 3 3 3 3 3 3 3
 3 3 1 3 1 3 1 1 1 3 3 4 3 3 3 3 3 1 2 1 3 5 3 6 1 3 4 3 5 1 4 1 3 6 4 3 3
 3 2 3 6 1 3 6 3 3 3 3 3 3 3 3 3 5 6 3 1 1 3 1 1 1 6 3 3 3 3 3 3 3 3 3 1 3
 1 6 3 3 3 1 4 1 3 1 3 3 1 3 4 5 3 1 3 6 6 3 3 3 4 3 3 1 3 3 3 3 1 3 3 3 3
 3 1 1 3 1 3 3 3 6 3 3 3 4 5 5 1 3 1 5 1 3 3 1 3 3 1 5 3 3 3 3 6 3 3 3 1 2
 3 3 3 3 2 6 3 2 3 6 3 3 3 3 3 3 3 3 3 5 1 4 2 6 3 1 5 3 3 4 3 3 1 3 3 6 5
 3 1 1 3 3 3 6 3 3 3 3 0 3 1 1 3 3 3 3 4 6 3 4 3 3 3 3 6 4 2 3 4 3 4 3 1 3
 3 3 3 3 1 3 6 6 1 6 1 6 1 3 3 6 3 3 3 3 3 1 1 3 3 3 1 3 3 3 4 3 3 5 3 3 3
 3 6 3 3 3 6 3 3 1 3 3 3 3 3 3 3 1 3 1 3 3 1 3 3 5 3]
Which Correct: [ True  True False  True  True  True False  True  True  True  True False
  True False  True  True  True  True  True  True  True False  True False
 False  True False  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True False  True
 False  True  True  True False  True False  True  True  True  True  True
 False  True False  True  True  True False  True  True  True False False
  True  True  True  True False  True False False False False False  True
  True  True  True  True  True  True False  True  True False  True False
  True  True False  True  True False  True False  True  True  True  True
  True  True False  True False  True  True False False  True  True  True
  True  True  True False  True  True False  True  True False  True  True
 False False False  True  True  True  True False False False False  True
  True False False  True False  True False False  True  True  True  True
  True False False  True False  True  True  True  True False  True  True
 False  True  True False False  True  True False  True False False False
  True  True  True False False  True  True  True  True  True False False
 False False  True  True  True False  True False  True False False False
 False False False  True False False False  True  True  True False  True
  True False False False False False False  True False  True  True  True
 False  True  True  True False  True  True  True False  True  True  True
  True  True False False False False  True  True  True  True  True False
 False False  True  True False  True False False False  True False  True
  True False False  True False False False  True  True  True  True  True
 False  True  True  True  True False  True  True False False  True  True
  True  True False  True  True False  True False False False  True False
  True  True False False  True  True False  True False  True  True  True
  True False False  True False  True  True  True  True False]
Total Correct: 199
Accuracy: 0.6180124223602484
                   precision    recall  f1-score   support

     Ariel Sharon       1.00      0.08      0.14        13
     Colin Powell       0.64      0.62      0.63        60
  Donald Rumsfeld       0.50      0.15      0.23        27
    George W Bush       0.64      0.88      0.74       146
Gerhard Schroeder       0.56      0.40      0.47        25
      Hugo Chavez       0.69      0.60      0.64        15
       Tony Blair       0.40      0.28      0.33        36

         accuracy                           0.62       322
        macro avg       0.63      0.43      0.45       322
     weighted avg       0.61      0.62      0.58       322
```