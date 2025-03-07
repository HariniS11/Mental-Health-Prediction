# Mental Health Prediction

## 📌 Project Overview
The **Mental Health Prediction** project aims to identify individuals at risk of depression based on various personal and lifestyle factors. By leveraging **deep learning**, this model analyzes input data to classify whether a person is likely to experience depression. The ultimate goal is to provide early insights that can assist in mental health awareness and intervention.

## 🚀 Objective
Mental health issues like depression are widespread and often go unnoticed until they become severe. This project utilizes **deep learning classification models** to analyze demographic and behavioral attributes, offering predictions that could help in early diagnosis and prevention strategies.

## 📂 Dataset Details
The dataset includes various demographic and lifestyle features:
- **Age** (Numerical)
- **Working Professional or Student** (Categorical - Binary Encoded)
- **Profession** (Categorical - Binary Encoded)
- **Sleep Duration** (Categorical - Label Encoded)
- **Degree** (Categorical - Binary Encoded)
- **Have you ever had suicidal thoughts?** (Categorical - Binary Encoded)
- **Work/Study Hours** (Categorical - Label Encoded)
- **Financial Stress** (Categorical - Label Encoded)
- **Depression (Target Variable)** (Binary: 0 = No Depression, 1 = Depression)

## 🏗️ Modeling Approach
This project uses a **deep learning-based classification model** with the following key steps:
1. **Data Preprocessing:**
   - **Label Encoding** applied to: `Work/Study Hours`, `Sleep Duration`, `Financial Stress`
   - **Binary Encoding** applied to: `Profession`, `Degree`, `Working Professional or Student`, `Suicidal Thoughts`
2. **Data Balancing:**
   - **Oversampling** technique (SMOTE) applied to handle class imbalance and improve model performance.
3. **Model Architecture:**
   - A deep learning neural network with multiple layers trained to classify individuals as at risk of depression or not.
4. **Evaluation Metrics:**
   - **Accuracy:** `90.02%`
   - **Precision:** `87.12%`
   - **Recall:** `93.91%`
   - **F1-Score:** `90.39%`

## 🌍 Deployment
- The model is deployed using **Streamlit Cloud**, allowing users to interact with the prediction system via a simple and user-friendly web interface.
- Users can enter their details, and the model will predict whether they are at risk of depression or not.

## 🎯 How to Use
1. **Visit the deployed Streamlit application.**
2. **Input Required Information:**
   - Age, Profession, Degree, Sleep Duration, Work/Study Hours, Financial Stress Level, and Suicidal Thoughts.
3. **Click 'Predict'**
4. **Get Instant Results:** The model will predict whether the person is at risk of depression or not.

## 📌 Key Takeaways
- This project demonstrates how **deep learning** can be used in mental health prediction.
- The model is designed to be **scalable and interpretable**, ensuring reliability in predictions.
- The **balanced dataset and feature encoding** significantly improve classification accuracy.
- **Deployment on Streamlit Cloud** ensures ease of access for users worldwide.

## 💡 Future Improvements
- **Feature Engineering:** Exploring additional features that may improve prediction accuracy.
- **Real-World Data Integration:** Expanding the dataset with real-world cases to improve generalization.
- **Model Optimization:** Fine-tuning hyperparameters for enhanced performance.

## 👨‍💻 Technologies Used
- **Python**
- **PyTorch** (Deep Learning Framework)
- **Scikit-learn** (Preprocessing & Encoding)
- **imbalanced-learn (SMOTE)**
- **Pandas & NumPy** (Data Handling)
- **Streamlit** (Deployment)

## 📜 License
This project is for educational and research purposes only. Proper mental health care should always be sought from qualified professionals.

---
🚀 **Built with passion to make mental health awareness accessible to all.**

