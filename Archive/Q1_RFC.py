data_path = "../../../Data_birds/birds_dataset/Bird Speciees Dataset/"
## Currently, this is accessing local file on sciserver
## To be updated to take dataset from Kaggle

## Geeks for Geeks

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import random
import numpy as np
##cv2 is tricky

##pip uninstall numpy scipy pandas bqplot
##pip install numpy==1.23.5 scipy pandas==1.5.3 bqplot==0.12.36

## pip install opencv-python==4.7.0.72
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

def extract_hog_features(image):
    hog_features = hog(
        image,
        orientations = 9,
        pixels_per_cell = (8, 8),
        cells_per_block = (2, 2),
        visualize=False
    )
    return hog_features

def load_and_extract_features(directory):
    x, y = [], []
    for label, class_name in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)

        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_resized = cv2.resize(img, (128,128))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            hog_features = extract_hog_features(img_gray)
            x.append(hog_features)
            y.append(label)
    return np.array(x), np.array(y)

x, y = load_and_extract_features("../../../Data_birds/birds_dataset/Bird Speciees Dataset/")

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

rf_clf = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs = 1,
    max_depth = 5
)

rf_clf.fit(x_train_scaled, y_train)

y_pred = rf_clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4}")