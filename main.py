import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2
import os
import joblib

# ==========================================================
# CONFIG
# ==========================================================

SPECTRUM_LENGTH = 500
MODEL_PATH = "spectral_model.keras"
SCALER_PATH = "scaler.save"

PRODUCTS = ["Milk", "Turmeric", "Honey"]

ADULTERANTS = {
    "Milk": "Urea / Melamine / Water",
    "Turmeric": "Cassava Starch / Chalk Powder",
    "Honey": "Sugar Syrup / HFCS"
}

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# HSV SPECTRUM EXTRACTION (STACKED IMAGE)
# ==========================================================

def extract_stacked_templates(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    h, w = mask.shape
    third = h // 3
    templates = {}

    for i, product in enumerate(PRODUCTS):

        roi = mask[int(i*third + third*0.15):
                   int((i+1)*third - third*0.10),
                   int(w*0.05):int(w*0.95)]

        roi_h, roi_w = roi.shape
        signal = []

        for x in range(roi_w):
            col = roi[:, x]
            pixels = np.where(col > 0)[0]
            if len(pixels) > 0:
                height = roi_h - np.mean(pixels)
                signal.append(height)
            else:
                signal.append(signal[-1] if len(signal) else 0)

        signal = np.array(signal, dtype=np.float32)
        if np.max(signal) > 0:
            signal /= np.max(signal)

        original_x = np.linspace(0,1,len(signal))
        target_x = np.linspace(0,1,SPECTRUM_LENGTH)

        templates[product] = np.interp(target_x, original_x, signal)

    return templates

# ==========================================================
# SINGLE IMAGE EXTRACTION
# ==========================================================

def extract_single_image(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    h, w = mask.shape
    roi = mask[int(h*0.15):int(h*0.85),
               int(w*0.05):int(w*0.95)]

    roi_h, roi_w = roi.shape
    signal = []

    for x in range(roi_w):
        col = roi[:, x]
        pixels = np.where(col > 0)[0]
        if len(pixels) > 0:
            height = roi_h - np.mean(pixels)
            signal.append(height)
        else:
            signal.append(signal[-1] if len(signal) else 0)

    signal = np.array(signal, dtype=np.float32)
    if np.max(signal) > 0:
        signal /= np.max(signal)

    original_x = np.linspace(0,1,len(signal))
    target_x = np.linspace(0,1,SPECTRUM_LENGTH)

    return np.interp(target_x, original_x, signal)

# ==========================================================
# DATA GENERATOR (PURE + CONTINUOUS ADULTERATION)
# ==========================================================

def generate_dataset(pure_templates, adulterated_templates, n=4000):

    X, y_product, y_concentration = [], [], []

    for _ in range(n):

        product_idx = np.random.randint(0, len(PRODUCTS))
        product = PRODUCTS[product_idx]

        pure = pure_templates[product]
        adulterated = adulterated_templates[product]

        is_adulterated = np.random.rand() > 0.5

        if is_adulterated:
            conc = np.random.uniform(0.05,1.0)
            spectrum = (1-conc)*pure + conc*adulterated
            concentration = conc * 100
        else:
            spectrum = pure.copy()
            concentration = 0.0

        spectrum += np.random.normal(0,0.015,SPECTRUM_LENGTH)
        spectrum = np.clip(spectrum,0,None)
        spectrum /= np.max(spectrum)

        X.append(spectrum)
        y_product.append(product_idx)
        y_concentration.append(concentration)

    return np.array(X), np.array(y_product), np.array(y_concentration)

# ==========================================================
# MODEL
# ==========================================================

def build_model():

    inputs = layers.Input(shape=(SPECTRUM_LENGTH,1))

    x = layers.Conv1D(32,5,activation='relu',padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64,5,activation='relu',padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu')(x)

    product_out = layers.Dense(len(PRODUCTS),activation='softmax',name="product")(x)
    concentration_out = layers.Dense(1,activation='linear',name="concentration")(x)

    model = keras.Model(inputs,[product_out,concentration_out])

    model.compile(
        optimizer='adam',
        loss={
            "product":"sparse_categorical_crossentropy",
            "concentration":"mse"
        },
        loss_weights={
            "product":1.0,
            "concentration":0.5
        },
        metrics={
            "product":"accuracy",
            "concentration":"mae"
        }
    )

    return model

# ==========================================================
# INFERENCE REPORT
# ==========================================================

def run_inference(model, scaler, pure_templates, spectrum):

    flat = spectrum.reshape(1,-1)
    scaled = scaler.transform(flat)
    processed = scaled.reshape(1,SPECTRUM_LENGTH,1)

    product_pred, conc_pred = model.predict(processed,verbose=0)

    product_idx = np.argmax(product_pred[0])
    product_name = PRODUCTS[product_idx]
    confidence = np.max(product_pred[0]) * 100

    concentration = float(np.clip(conc_pred[0][0],0,100))

    print("\n" + "="*65)
    print("FORENSIC ANALYSIS REPORT")
    print("="*65)
    print(f"Detected Product      : {product_name}")
    print(f"Classification Conf   : {confidence:.2f}%")
    print(f"Estimated Concentration: {concentration:.2f}%")

    if concentration < 50:
        print("STATUS                : ✅ PURE")
    else:
        print("STATUS                : 🚨 ADULTERATED")
        print(f"Likely Adulterant     : {ADULTERANTS[product_name]}")

    print("="*65)

    # Plot comparison
    reference = pure_templates[product_name]

    plt.figure(figsize=(12,4))
    plt.plot(reference,label="Pure Reference",linewidth=2)
    plt.plot(spectrum,label="Test Sample",linewidth=2)
    plt.legend()
    plt.title("Pure vs Test Spectrum Comparison")
    plt.grid(alpha=0.3)
    plt.show()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    PURE_IMAGE = "Pure samples.png"
    ADUL_IMAGE = "Adulterants.png"

    print("Extracting templates...")
    pure_templates = extract_stacked_templates(PURE_IMAGE)
    adulterated_templates = extract_stacked_templates(ADUL_IMAGE)

    print("Generating dataset...")
    X, y_product, y_concentration = generate_dataset(
        pure_templates,
        adulterated_templates,
        4000
    )

    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y_product, y_concentration,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1,SPECTRUM_LENGTH,1)
    X_test = X_test.reshape(-1,SPECTRUM_LENGTH,1)

    model = build_model()
    model.fit(
        X_train,
        {"product":y_train,"concentration":c_train},
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    model.save(MODEL_PATH)
    joblib.dump(scaler,SCALER_PATH)

    print("\nTraining Complete.")

    # ============================
    # GIVE YOUR TEST IMAGE HERE
    # ============================

    MY_IMAGE = "Milk adulterant.png"

    test_spectrum = extract_single_image(MY_IMAGE)

    run_inference(model, scaler, pure_templates, test_spectrum)
