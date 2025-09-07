import numpy as np
import scipy.io

def geneData(id):
    numOfStudent = 300
    index = int(id) % numOfStudent
    
    # Load image data
    try:
        Numpyfile0 = scipy.io.loadmat('data/train_0_img.mat')
        Numpyfile1 = scipy.io.loadmat('data/train_1_img.mat')
        train01 = Numpyfile0.get('target_img')
        train02 = Numpyfile1.get('target_img')
    except FileNotFoundError as e:
        print(f"Error loading training images: {e}")
        return

    # Load labels
    try:
        labels0 = scipy.io.loadmat('data/train_0_label.mat')['target_label']
        labels1 = scipy.io.loadmat('data/train_1_label.mat')['target_label']
    except FileNotFoundError as e:
        print(f"Error loading training labels: {e}")
        return
    
    train01 = np.transpose(train01, axes=[2, 0, 1])
    train02 = np.transpose(train02, axes=[2, 0, 1])
    
    np.random.seed(index)
    np.random.shuffle(train01)
    np.random.seed(index)
    np.random.shuffle(train02)
    
    newarr0 = train01[:5000]
    newarr1 = train02[:5000]
    newlabels0 = labels0[:5000]
    newlabels1 = labels1[:5000]
    
    # Save training data with labels
    scipy.io.savemat(f'digit0_stu_train{id}.mat', {'target_img': newarr0})
    scipy.io.savemat(f'digit1_stu_train{id}.mat', {'target_img': newarr1})
    scipy.io.savemat(f'digit0_stu_train{id}_label.mat', {'target_label': newlabels0})
    scipy.io.savemat(f'digit1_stu_train{id}_label.mat', {'target_label': newlabels1})

    # Load test data
    try:
        Numpyfile2 = scipy.io.loadmat('data/test_0_img.mat')
        Numpyfile3 = scipy.io.loadmat('data/test_1_img.mat')
        test01 = Numpyfile2.get('target_img')
        test02 = Numpyfile3.get('target_img')
    except FileNotFoundError as e:
        print(f"Error loading test images: {e}")
        return

    # Load test labels
    try:
        test_labels0 = scipy.io.loadmat('data/test_0_label.mat')['target_label']
        test_labels1 = scipy.io.loadmat('data/test_1_label.mat')['target_label']
    except FileNotFoundError as e:
        print(f"Error loading test labels: {e}")
        return
    
    test01 = np.transpose(test01, axes=[2, 0, 1])
    test02 = np.transpose(test02, axes=[2, 0, 1])
    
    # Save test data with labels
    scipy.io.savemat('digit0_testset.mat', {'target_img': test01})
    scipy.io.savemat('digit1_testset.mat', {'target_img': test02})
    scipy.io.savemat('digit0_testset_label.mat', {'target_label': test_labels0})
    scipy.io.savemat('digit1_testset_label.mat', {'target_label': test_labels1})
