# DSE
README for a dissertation titled ‘The Effect of Changing Sequences of Algorithms in Hybrid Anomaly Detection Methodology.’

Overview

This repository hosts the code and methodology for a PhD dissertation research project at the National University that compares hybrid anomaly detection models using a sequential approach. The dissertation focuses on exploring the impact of changing the sequence of algorithms in a sequential approach to anomaly detection. This involves analyzing how the order in which different algorithms are applied affects the overall effectiveness of anomaly detection systems. The sequential approach typically consists of using multiple algorithms in a specific order, where the output of one algorithm serves as the input for the next. By altering this sequence, the dissertation aims to identify optimal configurations that enhance the detection accuracy and efficiency of the system.
This type of research is crucial because the sequence in which algorithms are applied can significantly influence the performance of the anomaly detection system. Different sequences may influence how well the system identifies anomalies, affecting precision, recall, and F1 score metrics. The dissertation likely includes experiments where various sequences are tested and compared to determine which configurations yield the best results regarding accuracy and computational efficiency.
The research's emphasis on altering the sequence of algorithms underscores the practicality of not just selecting suitable algorithms for anomaly detection, but also arranging them in an order that optimizes their collective strengths while mitigating their weaknesses. This approach can pave the way for more robust anomaly detection systems that are better tailored for real-world applications across various domains.

Models Description
Sparse Autoencoder
The initial use of a Sparse Autoencoder is pivotal in this research. It reduces dimensionality and extracts pertinent features from the dataset. This step is critical as it bolsters the performance of the subsequent models by focusing on the most informative features.
Support Vector Machine (SVM)
SVM is a well-known algorithm for anomaly detection, especially effective in high-dimensional spaces. It is used here as a baseline to compare the performance of other models.
Isolation Forest
Isolation Forest is an algorithm particularly effective for identifying anomalies in the data. It isolates anomalies instead of profiling normal data points.
XGBoost
In some instances, XGBoost is introduced as an alternative to SVM. It is known for its high performance in classification tasks, especially with structured data.

Sequential Approach and Model Interplay
The models are applied sequentially, with the Sparse Autoencoder first processing the data to extract features. These features are then fed into the SVM, Isolation Forest, and XGBoost models. The sequence in which these models are applied and the replacement of SVM with XGBoost in some scenarios are analyzed to understand their impact on anomaly detection performance.

Evaluation Metrics
Precision, Recall, and F1 Score
•	Precision: Measures the accuracy of optimistic predictions.
•	Recall: Measures the ability of the model to find all the relevant cases (all anomalies).
•	F1 Score: Harmonic mean of precision and recall, providing a balance between them.

Confusion Matrix
The confusion matrix is a valuable tool for understanding the performance of classification models. It provides insight into the types of errors made by the models.

Dataset
The datasets employed in this project are secondary and pulled from publicly available websites. They are chosen to represent a variety of scenarios in anomaly detection and test the models' robustness without relying on domain knowledge.

Installation and Usage
Prerequisites
•	Python 3.8+
•	NumPy
•	Scikit-learn.
•	XGBoost
•	TensorFlow or another backend supporting Keras.

Setup
To set up the project environment:
git clone https://github.com/sebhatgg/DSE

Install the required packages:
pip install -r requirements.txt

Running the Models
To run the anomaly detection models, execute:
python ‘modelName’.py

Results
The results, including precision, recall, F1 scores, and confusion matrices, will be output to the console and saved in the results directory for further analysis and comparison.

Dissertation Use
This repository is part of a PhD dissertation at the National University. The code and findings are intended for academic purposes and are not licensed by any organization.

Acknowledgments
•	The author would like to thank the National University for its support during the research.
•	Appreciation goes to the open-source community for the development of the machine-learning libraries used in this project.

This README provides the necessary information to understand and engage with the hybrid anomaly detection models presented in this repository.
