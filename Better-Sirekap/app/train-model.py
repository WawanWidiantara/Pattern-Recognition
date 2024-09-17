import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.pipeline import Pipeline
import pickle


# Define preprocessing and feature extraction functions
def preprocess_and_crop_image(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped = gray[y : y + h, x : x + w]
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        return resized
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return resized


def extract_hog_features(image):
    features, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
    )
    return features


def extract_zoning_features(image, zones=(4, 4)):
    h, w = image.shape
    zh, zw = h // zones[0], w // zones[1]
    features = []
    for i in range(zones[0]):
        for j in range(zones[1]):
            zone = image[i * zh : (i + 1) * zh, j * zw : (j + 1) * zw]
            features.append(np.mean(zone))
    return np.array(features)


def extract_pca_features(image, pca):
    flattened = image.flatten().reshape(1, -1)
    pca_features = pca.transform(flattened)
    return pca_features.flatten()


def extract_combined_features(image_path, pca):
    image = preprocess_and_crop_image(image_path)
    hog_features = extract_hog_features(image)
    zoning_features = extract_zoning_features(image)
    pca_features = extract_pca_features(image, pca)
    return np.concatenate([hog_features, zoning_features, pca_features])


# Directory containing the image data
data_dir = r"D:\Code\py_code\Pattern-Recognition\Better-Sirekap\data"
img_list = os.listdir(data_dir)
img_files = [os.path.join(data_dir, img) for img in img_list]

# Label extraction from image filenames
data = img_files
labels = [int(img[0]) for img in img_list]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Prepare PCA on training data
all_images = [preprocess_and_crop_image(img) for img in X_train]
flattened_images = [img.flatten() for img in all_images]
pca = PCA(n_components=min(50, len(flattened_images[0])), random_state=42)
pca.fit(flattened_images)

# Feature extraction combinations
feature_extraction_combinations = [
    ("HOG + Zoning + PCA", lambda img: extract_combined_features(img, pca)),
    ("HOG", lambda img: extract_hog_features(preprocess_and_crop_image(img))),
    ("Zoning", lambda img: extract_zoning_features(preprocess_and_crop_image(img))),
    ("PCA", lambda img: extract_pca_features(preprocess_and_crop_image(img), pca)),
    (
        "HOG + Zoning",
        lambda img: np.concatenate(
            [
                extract_hog_features(preprocess_and_crop_image(img)),
                extract_zoning_features(preprocess_and_crop_image(img)),
            ]
        ),
    ),
    (
        "HOG + PCA",
        lambda img: np.concatenate(
            [
                extract_hog_features(preprocess_and_crop_image(img)),
                extract_pca_features(preprocess_and_crop_image(img), pca),
            ]
        ),
    ),
    (
        "Zoning + PCA",
        lambda img: np.concatenate(
            [
                extract_zoning_features(preprocess_and_crop_image(img)),
                extract_pca_features(preprocess_and_crop_image(img), pca),
            ]
        ),
    ),
    ("Raw Pixels", lambda img: preprocess_and_crop_image(img).flatten()),
]

# Classifier configurations
classifiers = [
    ("SVM", SVC(kernel="linear", random_state=42, gamma=0.001, C=10, probability=True)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
]

# Train and save models
for feature_name, feature_func in feature_extraction_combinations:
    X_train_features = np.array([feature_func(img) for img in X_train])
    X_test_features = np.array([feature_func(img) for img in X_test])

    # Scale the features
    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    for clf_name, clf in classifiers:
        clf.fit(X_train_features, y_train)

        # Create a pipeline
        pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])

        # Save the pipeline and PCA
        model_filename = f"{feature_name.replace(' ', '_')}_{clf_name}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(
                {
                    "pipeline": pipeline,
                    "pca": pca if "PCA" in feature_name else None,
                },
                file,
            )
