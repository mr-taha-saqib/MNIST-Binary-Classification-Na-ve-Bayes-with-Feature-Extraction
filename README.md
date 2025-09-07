# MNIST Binary Classification: Naïve Bayes with Feature Extraction

## 📌 Project Description
This project involves implementing **supervised, unsupervised, and deep learning techniques** for density estimation and classification.  

The project focuses on a subset of the **MNIST dataset** containing images of digits **"0"** and **"1"**.  

It consists of four main tasks:  
1. Feature extraction  
2. Parameter calculation  
3. Implementation of Naïve Bayes classifiers  
4. Prediction of labels for the test data using the classifiers and calculation of accuracy  

---

## 🛠 Preparation
In the `Project1` Jupyter notebook, you will:  
- Load the **trainset** and **testset** for digit **0** and digit **1** respectively.  
- Both trainset and testset are sub-datasets extracted from MNIST.  

### Dataset Details
- **Training set**:  
  - Digit "0": 5000 samples  
  - Digit "1": 5000 samples  
- **Testing set**:  
  - Digit "0": 980 samples  
  - Digit "1": 1135 samples  

We assume equal prior probabilities:  
P(Y=0) = P(Y=1) = 0.5


### Variables in Existing Code
- `myID`: A 4-digit string (student ID).  
- `train0`, `train1`: Trainset arrays for digit 0 and digit 1.  
- `test0`, `test1`: Testset arrays for digit 0 and digit 1.  

All are **NumPy arrays** (can be converted into Python arrays if needed).  

### Reading the Dataset
The dataset has also been stored in `.mat` files for convenience:  
