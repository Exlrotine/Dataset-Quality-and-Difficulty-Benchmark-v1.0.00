import os
import random
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from data_loader import data_from_folder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler   # SVM 建议先标准化
from sklearn.model_selection import GridSearchCV   # 这个搜索非常消耗内存，可以一次少搜一点


# Function to extract features from an image
def extract_features(image_path, target_size=(200, 200)):
    try:
        img = Image.open(image_path)    # .convert('L')  # Convert to grayscale
        img = img.resize(target_size)  # Resize to reduce dimensionality
        img_array = np.array(img).flatten()
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Load data
all_data = data_from_folder()
X = []
y = []
for img_path, class_name in all_data:
    features = extract_features(img_path)
    if features is not None:
        X.append(features)
        y.append(class_name)
X = np.array(X)
y = np.array(y)

# Encode labels (convert class names to integers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
pca = PCA(n_components=0.95, svd_solver='randomized', whiten=True, random_state=42)
rf_pipe = Pipeline([('pca', pca), ('rf', RandomForestClassifier(random_state=42))])


# rf_search = RandomForestClassifier(n_estimators=500, random_state=42)
param_grid_rf = {'pca__n_components': [50], 'rf__n_estimators': [100, 300, 500],}
rf_search = GridSearchCV(rf_pipe, param_grid=param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
rf_search.fit(X_train, y_train)
print("最佳 RF 组合：", rf_search.best_params_, "CV-Acc=", rf_search.best_score_)
y_pred_rf = rf_search.best_estimator_.predict(X_test)
# y_pred_rf = rf_search.predict(X_test)
print("random forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# svm_search = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_pipe = Pipeline([('scale', StandardScaler()), ('pca', pca), ('svm', SVC(kernel='rbf', random_state=42))])
param_grid_svm = {'pca__n_components': [30], 'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale']}
svm_search = GridSearchCV(svm_pipe, param_grid=param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
svm_search.fit(X_train, y_train)
print("最佳 SVM 组合：", svm_search.best_params_, "CV-Acc=", svm_search.best_score_)
y_pred_svm = svm_search.best_estimator_.predict(X_test)
# y_pred_svm = svm_search.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

