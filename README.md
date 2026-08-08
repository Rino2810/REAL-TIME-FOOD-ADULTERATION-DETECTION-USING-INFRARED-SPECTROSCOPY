
 REAL-TIME FOOD ADULTERATION DETECTION USING INFRARED SPECTROSCOPY



This repository contains the code and methodology for a portable, non-destructive hybrid system designed for the real-time detection of food adulterants. By synergizing Infrared Spectroscopy, the project utilizes a lightweight 1D-CNN to deliver rapid diagnostic feedback directly in the field, eliminating the need for cloud connectivity.


## Problem Statement

* **Slow & Costly Lab Testing:** Accurate methods like chromatography are expensive, confined to labs, and have long turnaround times.


* **Lack of Accessibility:** Testing requires complex equipment and highly skilled technicians, creating barriers for consumers and small vendors.


* **Poor Generalization:** Most chemical kits only work for specific food types and a limited set of adulterants.


* **Limited Dataset Availability:** There is a scarcity of annotated spectral datasets for food products, hindering robust ML model development.



## Project Objectives

* Develop an efficient, portable hardware-software system suitable for real-time field deployment.


* Ensure robust detection across diverse food products like milk, turmeric, and honey.


* Achieve state-of-the-art accuracy in classifying samples as "Pure" or "Adulterated".


* Provide a user-friendly interface with immediate outputs, removing the need for expert interpretation.



## Methodology

### Module 1: Signal Extraction & Preprocessing

* **HSV Color Isolation:** Uses OpenCV to convert images into HSV color space, isolating specific plot lines from templates while ignoring backgrounds.


* **Energy-Based Signal Capture:** Calculates the height of spectral waves to convert visual information into a raw 1D digital signal.


* **Spectral Normalization:** Scales all signals to a 500-band length and processes them via a Standard Scaler for uniform neural network input.



### Module 2: Empirical Data Generation

* **Continuous Concentration Blending:** Dynamically mixes pure and adulterated templates at random ratios (5% to 100%) rather than using binary labels.


* **Robustness Training:** Adds baseline shifts and Gaussian noise to synthetic samples to simulate real-world sensor fluctuations.


* **Dataset Balancing:** Creates a balanced dataset of 4,000 samples (3,200 Train / 800 Test) to help the AI learn unique spectral fingerprints.



### Module 3: Multi-Output 1D-CNN Architecture

* **Shared Feature Extraction:** Dual Conv1D layers with Batch Normalization and MaxPooling extract deep patterns from spectral waves.


* **Classification Branch:** A Softmax head identifies the base food product and provides a confidence percentage.


* **Regression Branch:** A ReLU-activated head calculates the exact concentration of suspected chemical contaminants (e.g., urea, sugar syrup).



## Dataset

* **Type:** Empirical Spectral Data extracted from stacked image templates.


* **Base Products:** Milk, Turmeric, Honey.


* **Adulterants:** Urea/Melamine/Water, Cassava Starch/Chalk Powder, Sugar Syrup/HFCS.



## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/username/food-adulteration-detection.git
cd food-adulteration-detection

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Execute the main training pipeline:
```bash
python main.py

```



## Future Scope

* Integrate physical spectral sensors with the refined AI model.


* Expand detection capabilities across diverse commodities and environmental conditions for robust deployment.


* Develop a desktop-based quality control tool for field inspections.
