# 🔢 MNIST Digit Recognition Using Machine Learning

## 📌 Project Overview

**MNIST Digit Recognition** is a Machine Learning project that recognizes handwritten digits from **0 to 9** using image pixel data.

The project uses the **MNIST handwritten digit dataset**, where each digit image is represented by **784 pixel values** corresponding to a **28 × 28 grayscale image**. The data is pre-processed, visualized, normalized, and then given to a Machine Learning model for classification.

The main goal is to build a model that can identify handwritten digits and evaluate its predictions using different classification metrics and visualizations.

---

## 🎯 Objectives

* Load and explore the MNIST dataset.
* Understand the structure and distribution of digit images.
* Check and handle missing values.
* Visualize handwritten digit images.
* Analyze pixel intensity values.
* Normalize image pixel values.
* Split the dataset into training and testing sets.
* Train a Support Vector Machine (SVM) classifier.
* Predict handwritten digits from unseen test data.
* Evaluate the model using accuracy, precision, recall, and F1-score.
* Analyze classification errors using a confusion matrix.
* Visualize correct and incorrect predictions.

---

## 📊 Dataset

The project uses the **MNIST in CSV** dataset available on Kaggle.

[Kaggle MNIST in CSV Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?utm_source=chatgpt.com)

### Dataset Information

| Feature          | Description              |
| ---------------- | ------------------------ |
| Dataset          | MNIST Handwritten Digits |
| Image Size       | 28 × 28 pixels           |
| Pixel Features   | 784                      |
| Classes          | 10                       |
| Classes Range    | 0–9                      |
| Image Type       | Grayscale                |
| Target Column    | `label`                  |
| Training Samples | 60,000                   |

Each image contains 784 pixel values ranging from **0 to 255**.

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Support Vector Machine (SVM)
* Jupyter Notebook / Kaggle Notebook

---

# 🔄 Project Workflow

```text
MNIST Dataset
      ↓
Load Dataset
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Separate Features & Labels
      ↓
Data Visualization
      ↓
Pixel Normalization
      ↓
Train-Test Split
      ↓
Train SVM Model
      ↓
Make Predictions
      ↓
Evaluate Model
      ↓
Confusion Matrix
      ↓
Analyze Correct & Incorrect Predictions
```

---

# 📁 Project Steps

## 1. Import Libraries

The required Python libraries are imported for:

* Data processing
* Numerical operations
* Data visualization
* Machine Learning
* Model evaluation

---

## 2. Load the Dataset

The `mnist_train.csv` file is loaded using Pandas.

```python
data = pd.read_csv("mnist_train.csv")
```

### Observation

The MNIST dataset is successfully loaded into a Pandas DataFrame.

---

## 3. Explore Dataset Shape and Data

The dataset structure is examined using:

```python
data.shape
data.head()
```

### Observation

The dataset contains handwritten digit records along with their corresponding pixel values.

---

## 4. Dataset Information

The structure and data types of the dataset are checked using:

```python
data.info()
```

### Observation

The dataset contains one target column, `label`, and 784 pixel-feature columns.

---

## 5. Check Missing Values

Missing values are checked using:

```python
data.isnull().sum().sum()
```

### Observation

No missing values were found in the dataset, so no missing-value treatment was required.

---

# 🏷️ 6. Separate Features and Target

The `label` column is used as the target, while the remaining columns are used as input features.

```python
X = data.drop("label", axis=1)
y = data["label"]
```

### Features

The 784 pixel columns represent the handwritten digit image.

### Target

The `label` column contains the actual digit from **0 to 9**.

### Observation

The dataset is successfully divided into input features and target labels.

---

# 📊 7. Analyze Digit Distribution

The number of samples for each digit is visualized using a bar chart.

### Purpose

This helps understand how the handwritten digit classes are distributed in the dataset.

### Observation

The graph shows the number of available samples for digits **0 through 9**.

---

# 🖼️ 8. Visualize Handwritten Digits

Several handwritten digit images are displayed by reshaping the 784 pixel values into a 28 × 28 image.

```text
784 Pixels
    ↓
28 × 28
    ↓
Grayscale Image
```

### Observation

The images show different handwriting styles for the digits 0–9.

---

# 📈 9. Pixel Intensity Analysis

A histogram is used to analyze pixel intensity values.

### Pixel Range

```text
0   → Black
255 → White
```

The pixel values represent the intensity of each pixel in the grayscale image.

### Observation

Most pixels represent the background of the image, while higher-intensity pixels generally represent parts of the handwritten digit.

---

# ⚙️ 10. Normalize Pixel Values

The pixel values are originally between **0 and 255**.

They are normalized to a range between **0 and 1**:

```python
X = X / 255.0
```

### Why Normalization?

Normalization makes the feature values smaller and puts them on a consistent scale, which can help the Machine Learning model work more effectively.

### Observation

All pixel values are converted from the range **0–255** to **0–1**.

---

# ✂️ 11. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Dataset Split

```text
80% → Training Data
20% → Testing Data
```

### Purpose

* **Training data:** Used to learn digit patterns.
* **Testing data:** Used to evaluate the model on unseen images.

### Observation

The dataset is successfully divided into training and testing sets.

---

# 🤖 12. Train the SVM Model

A **Support Vector Machine (SVM)** classifier is used for handwritten digit classification.

```python
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)
```

### Model Configuration

| Parameter | Value                  |
| --------- | ---------------------- |
| Algorithm | Support Vector Machine |
| Kernel    | RBF                    |
| C         | 10                     |
| Gamma     | Scale                  |
| Classes   | 0–9                    |

### Why SVM?

SVM is a supervised Machine Learning algorithm that can classify data into multiple classes. It is suitable for classification problems such as handwritten digit recognition.

### Observation

The SVM model learns patterns from the training images and their corresponding digit labels.

---

# 🔮 13. Make Predictions

The trained model is used to predict the labels of the test images.

```python
y_pred = model.predict(X_test)
```

### Observation

The model generates a predicted digit for every image in the testing dataset.

---

# 🎯 14. Model Accuracy

Accuracy is calculated using:

```python
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

### Formula

```text
Accuracy =
Correct Predictions
------------------- × 100
Total Predictions
```

### Observation

The accuracy value represents the percentage of test images correctly classified by the model.

> **Note:** The final accuracy should be taken from the output of your own notebook execution.

---

# 📋 15. Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
print(classification_report(y_test, y_pred))
```

### Purpose

This provides a detailed evaluation of how well the model performs for each digit from **0 to 9**.

### Observation

The classification report allows the performance of individual digit classes to be examined.

---

# 🔲 16. Confusion Matrix

A confusion matrix is created to understand classification errors.

```python
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

disp.plot(cmap="Blues", values_format="d")
plt.title("MNIST Digit Recognition - Confusion Matrix")
plt.show()
```

### Interpretation

```text
Diagonal Values
      ↓
Correct Predictions

Off-Diagonal Values
      ↓
Incorrect Predictions
```

### Observation

The confusion matrix shows which digits are correctly classified and which digits are confused with other digits.

---

# 🖼️ 17. Actual vs Predicted Digits

The project compares the actual digit with the model's predicted digit.

Example:

```text
Actual:    7
Predicted: 7
```

### Purpose

This visualization makes it easier to understand the model's predictions on individual handwritten images.

### Observation

Correct predictions occur when the actual and predicted labels are the same.

---

# ✅ 18. Correct Predictions Analysis

The number of correctly classified images for each digit is visualized.

### Purpose

This helps identify how many test images from each digit class were successfully recognized.

### Observation

The graph provides a class-wise view of correctly recognized handwritten digits.

---

# ❌ 19. Incorrect Predictions

The project calculates the number of incorrect predictions.

```text
Incorrect Predictions =
Total Test Samples - Correct Predictions
```

Incorrectly classified images are also displayed for further analysis.

### Observation

The incorrectly classified images help identify handwritten digits that are difficult for the model to distinguish.

---

# 🔍 20. Analyze Misclassified Images

Incorrectly classified images are displayed along with their actual and predicted labels.

Example:

```text
Actual:    4
Predicted: 9
```

### Purpose

This helps understand why some handwritten digits may be difficult to classify because of similarities in handwriting.

---

# 📊 21. Incorrect Predictions by Digit

A graph is used to show the number of incorrect predictions for each digit.

### Observation

The graph identifies the digit classes where classification errors occurred more frequently.

---

# 🔎 22. Individual Digit Prediction

The project also tests an individual image from the test dataset.

The image is displayed together with:

```text
Actual Digit
Predicted Digit
```

### Observation

This provides a simple visual demonstration of the trained model's digit recognition capability.

---

# 📈 Model Evaluation

The model is evaluated using multiple methods:

| Evaluation Method  | Purpose                              |
| ------------------ | ------------------------------------ |
| Accuracy           | Overall correct predictions          |
| Precision          | Correct positive predictions         |
| Recall             | Correctly identified samples         |
| F1-Score           | Balance between precision and recall |
| Confusion Matrix   | Class-wise prediction errors         |
| Visual Predictions | Direct image-level comparison        |

---

# 🔑 Key Project Highlights

* Handwritten digit recognition from **0 to 9**
* 28 × 28 grayscale image processing
* 784 pixel features
* Data exploration and visualization
* Pixel normalization
* Stratified train-test splitting
* SVM-based classification
* Accuracy evaluation
* Classification report
* Confusion matrix analysis
* Correct prediction analysis
* Misclassified image analysis
* Individual image prediction

---

# 📂 Project Structure

```text
MNIST-Digit-Recognition/
│
├── mnist_train.csv
├── MNIST_Digit_Recognition.ipynb
└── README.md
```

---

# 🚀 How to Run the Project

### 1. Download the Dataset

Download the MNIST CSV dataset from Kaggle.

[Download MNIST in CSV Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?utm_source=chatgpt.com)

### 2. Place the Dataset

Place:

```text
mnist_train.csv
```

in the same directory as the Jupyter Notebook.

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Run the Notebook

Open:

```text
MNIST_Digit_Recognition.ipynb
```

Run the cells from top to bottom.

---

# 💡 Applications

Handwritten digit recognition is a fundamental Computer Vision and Machine Learning problem and can be used as a foundation for applications such as:

* Handwritten form processing
* Digitized documents
* Postal code recognition
* Bank cheque processing
* Optical Character Recognition (OCR)
* Educational handwriting systems
* Automated number recognition

---

# 🧠 Learning Outcomes

Through this project, the following concepts were practiced:

* Dataset loading
* Data exploration
* Data preprocessing
* Feature and target selection
* Image visualization
* Pixel normalization
* Train-test splitting
* Supervised Machine Learning
* SVM classification
* Model prediction
* Classification metrics
* Confusion matrix
* Error analysis
* Data visualization

---

# 📝 Conclusion

The MNIST Digit Recognition project demonstrates how Machine Learning can be applied to handwritten image classification. The project follows a complete workflow from **dataset loading and preprocessing to model training, prediction, evaluation, and error analysis**.

Using pixel-based image features and an SVM classifier, the system learns patterns from handwritten digits and predicts the corresponding class from **0 to 9**. The visualizations and evaluation metrics provide a clear understanding of the model's classification performance and prediction errors.

---

# 👨‍💻 Author

**Zarfshan Attiq Khan**

**zarfshankhan478@gmail.com**