import cv2
import numpy as np
import streamlit as st
import pickle
from skimage.feature import hog


# Preprocess and crop the image to the target size
def preprocess_and_crop_image(image, target_size=(48, 48)):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

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


def extract_combined_features(image, pca):
    hog_features = extract_hog_features(image)
    zoning_features = extract_zoning_features(image)
    pca_features = extract_pca_features(image, pca)
    return np.concatenate([hog_features, zoning_features, pca_features])


# Streamlit interface
st.title("Image Classification with Feature Extraction and Model Selection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    with col2:
        feature_extraction_method = st.selectbox(
            "Select Feature Extraction Method",
            [
                "HOG + Zoning + PCA",
                "HOG",
                "Zoning",
                "PCA",
                "HOG + Zoning",
                "HOG + PCA",
                "Zoning + PCA",
                "Raw Pixels",
            ],
        )

        model_selection = st.selectbox("Select Model", ["SVM", "Random Forest"])

        predict_button = st.button("Predict")

    if predict_button:
        model_filename = (
            f"{feature_extraction_method.replace(' ', '_')}_{model_selection}.pkl"
        )

        with open(model_filename, "rb") as file:
            model_data = pickle.load(file)

        pipeline = model_data["pipeline"]
        pca = model_data["pca"]

        if feature_extraction_method == "HOG + Zoning + PCA":
            feature_func = lambda img: extract_combined_features(
                preprocess_and_crop_image(img), pca
            )
        elif feature_extraction_method == "HOG":
            feature_func = lambda img: extract_hog_features(
                preprocess_and_crop_image(img)
            )
        elif feature_extraction_method == "Zoning":
            feature_func = lambda img: extract_zoning_features(
                preprocess_and_crop_image(img)
            )
        elif feature_extraction_method == "PCA":
            feature_func = lambda img: extract_pca_features(
                preprocess_and_crop_image(img), pca
            )
        elif feature_extraction_method == "HOG + Zoning":
            feature_func = lambda img: np.concatenate(
                [
                    extract_hog_features(preprocess_and_crop_image(img)),
                    extract_zoning_features(preprocess_and_crop_image(img)),
                ]
            )
        elif feature_extraction_method == "HOG + PCA":
            feature_func = lambda img: np.concatenate(
                [
                    extract_hog_features(preprocess_and_crop_image(img)),
                    extract_pca_features(preprocess_and_crop_image(img), pca),
                ]
            )
        elif feature_extraction_method == "Zoning + PCA":
            feature_func = lambda img: np.concatenate(
                [
                    extract_zoning_features(preprocess_and_crop_image(img)),
                    extract_pca_features(preprocess_and_crop_image(img), pca),
                ]
            )
        elif feature_extraction_method == "Raw Pixels":
            feature_func = lambda img: preprocess_and_crop_image(img).flatten()

        preprocessed_image = preprocess_and_crop_image(image)
        features = feature_func(preprocessed_image)
        features = features.reshape(1, -1)

        y_pred = pipeline.predict(features)

        st.write(f"Prediction Label: {y_pred[0]}")

        st.image(
            preprocessed_image, caption="Preprocessed Image", use_column_width=True
        )
