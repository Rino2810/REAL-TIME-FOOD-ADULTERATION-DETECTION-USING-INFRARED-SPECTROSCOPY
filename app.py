import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

# ==========================================================
# CONFIG
# ==========================================================
SPECTRUM_LENGTH = 500
PRODUCTS = ["Milk", "Turmeric", "Honey"]
ADULTERANTS = {
    "Milk": "Urea / Melamine / Water",
    "Turmeric": "Cassava Starch / Chalk Powder",
    "Honey": "Sugar Syrup / HFCS"
}

# ==========================================================
# SYSTEM SETUP & LOGIC
# ==========================================================
def load_system_artifacts():
    try:
        # UPDATED: Now points to the modern .keras format
        model = load_model("spectral_model.keras")
        scaler = joblib.load("scaler.save")

        img = cv2.imread("Pure samples.png")
        if img is None:
            raise FileNotFoundError("Could not find 'Pure samples.png'.")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 40, 50])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        h, w = mask.shape
        third = h // 3
        templates = {}

        for i, product in enumerate(PRODUCTS):
            roi = mask[int(i*third + third*0.15):int((i+1)*third - third*0.10), int(w*0.05):int(w*0.95)]
            roi_h, roi_w = roi.shape
            signal = []
            for x in range(roi_w):
                col = roi[:, x]
                pixels = np.where(col > 0)[0]
                if len(pixels) > 0:
                    signal.append(roi_h - np.mean(pixels))
                else:
                    signal.append(signal[-1] if len(signal) else 0)

            signal = np.array(signal, dtype=np.float32)
            if np.max(signal) > 0:
                signal /= np.max(signal)

            templates[product] = np.interp(np.linspace(0, 1, SPECTRUM_LENGTH), np.linspace(0, 1, len(signal)), signal)

        return model, scaler, templates
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Failed to load required files:\n{e}")
        return None, None, None

def extract_single_image(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    h, w = mask.shape
    roi = mask[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]
    roi_h, roi_w = roi.shape
    signal = []

    for x in range(roi_w):
        col = roi[:, x]
        pixels = np.where(col > 0)[0]
        if len(pixels) > 0:
            signal.append(roi_h - np.mean(pixels))
        else:
            signal.append(signal[-1] if len(signal) else 0)

    signal = np.array(signal, dtype=np.float32)
    if np.max(signal) > 0:
        signal /= np.max(signal)

    return np.interp(np.linspace(0, 1, SPECTRUM_LENGTH), np.linspace(0, 1, len(signal)), signal)

# ==========================================================
# TKINTER GUI CLASS
# ==========================================================
class ForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Food Forensics")
        self.root.geometry("900x750")

        # Load backend
        self.model, self.scaler, self.pure_templates = load_system_artifacts()

        # Top Frame: Controls
        top_frame = tk.Frame(root, pady=10)
        top_frame.pack(fill=tk.X)

        tk.Label(top_frame, text="Micro-Spectral Food Forensics", font=("Arial", 16, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Button(top_frame, text="Upload Image & Analyze", command=self.upload_and_analyze, font=("Arial", 12), bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=20)

        # Middle Frame: Image and Results
        mid_frame = tk.Frame(root, pady=10)
        mid_frame.pack(fill=tk.X, padx=20)

        # Image Display
        self.img_label = tk.Label(mid_frame, text="No Image Uploaded", bg="gray", width=40, height=15)
        self.img_label.pack(side=tk.LEFT, padx=10)

        # Results Display
        res_frame = tk.Frame(mid_frame)
        res_frame.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)

        self.lbl_product = tk.Label(res_frame, text="Product: ---", font=("Arial", 14))
        self.lbl_product.pack(anchor="w", pady=5)

        self.lbl_conf = tk.Label(res_frame, text="Confidence: ---", font=("Arial", 14))
        self.lbl_conf.pack(anchor="w", pady=5)

        self.lbl_conc = tk.Label(res_frame, text="Adulteration: ---", font=("Arial", 14))
        self.lbl_conc.pack(anchor="w", pady=5)

        self.lbl_status = tk.Label(res_frame, text="Status: ---", font=("Arial", 14, "bold"))
        self.lbl_status.pack(anchor="w", pady=5)

        self.lbl_adulterant = tk.Label(res_frame, text="", font=("Arial", 12), fg="red")
        self.lbl_adulterant.pack(anchor="w", pady=5)

        # Bottom Frame: Plot
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def upload_and_analyze(self):
        if not self.model:
            messagebox.showerror("Error", "Models not loaded. Cannot analyze.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        # 1. Update UI Image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.configure(image=img_tk, text="")
        self.img_label.image = img_tk # Keep reference

        # 2. Run Inference
        try:
            test_spectrum = extract_single_image(file_path)
            flat = test_spectrum.reshape(1, -1)
            scaled = self.scaler.transform(flat)
            processed = scaled.reshape(1, SPECTRUM_LENGTH, 1)

            product_pred, conc_pred = self.model.predict(processed, verbose=0)

            product_idx = np.argmax(product_pred[0])
            product_name = PRODUCTS[product_idx]
            confidence = np.max(product_pred[0]) * 100
            concentration = float(np.clip(conc_pred[0][0], 0, 100))

            # 3. Update Labels
            self.lbl_product.config(text=f"Product: {product_name}")
            self.lbl_conf.config(text=f"Confidence: {confidence:.2f}%")
            self.lbl_conc.config(text=f"Adulteration: {concentration:.2f}%")

            if concentration < 10:
                self.lbl_status.config(text="Status: ✅ PURE", fg="green")
                self.lbl_adulterant.config(text="")
            else:
                self.lbl_status.config(text="Status: 🚨 ADULTERATED", fg="red")
                self.lbl_adulterant.config(text=f"Likely Adulterant: {ADULTERANTS[product_name]}")

            # 4. Update Plot
            self.ax.clear()
            self.ax.plot(self.pure_templates[product_name], label="Pure Reference", color="blue")
            self.ax.plot(test_spectrum, label="Test Sample", color="red", alpha=0.8)
            self.ax.legend()
            self.ax.grid(alpha=0.3)
            self.ax.set_title("Spectrum Comparison")
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{e}")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ForensicsApp(root)
    root.mainloop()