import streamlit as st
import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np

# Label encoders
with open("Work_Study_le.pkl", "rb") as f:
    le_work_study = pickle.load(f)

with open("Sleep_Duration_le.pkl", "rb") as f:
    le_sleep = pickle.load(f)

with open("Financial_Stress_le.pkl", "rb") as f:
    le_financial = pickle.load(f)

# Binary encode
with open('pro_encoder.pkl','rb') as f:
    pro_encoder = pickle.load(f)

with open('deg_encoder.pkl','rb') as f:
    deg_encoder = pickle.load(f)

with open('w_s_encoder.pkl','rb') as f:
    w_s_encoder = pickle.load(f)

with open('thoughts_encoder.pkl','rb') as f:
    thoughts_encoder = pickle.load(f)

    # Age scaling    
with open('age_scaler.pkl','rb') as f:
    age_scaler = pickle.load(f) 

class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.layer1 = nn.Linear(22, 48)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(48, 32)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Load Model
clf = classification()
clf.load_state_dict(torch.load("classification_weights.pth"))
clf.eval()

df = pd.read_csv('train.csv')
valid_professions = df["Profession"].dropna().unique().tolist()
valid_sleep_duration = df["Sleep Duration"].dropna().unique().tolist()
valid_degree = df['Degree'].dropna().unique().tolist()
work_study_hrs = sorted(df['Work/Study Hours'].dropna().unique().tolist())
financial_stress = sorted(df['Financial Stress'].dropna().unique().tolist())

financial_stress_mapping = {
    5.0: "High Financial Stress",
    4.0: "Below High",
    3.0: "Moderate",
    2.0: "Low",
    1.0: "Very Low"
}


verbal_to_numeric = {v: k for k, v in financial_stress_mapping.items()}

st.set_page_config(page_title="Mental Health Care App", layout="wide")

page = st.sidebar.radio("Navigation", ["Welcome", 'Health Prediction'])

if page == "Welcome":
    st.title("Mental Health Care App")
    st.subheader('HEALTH IS WEALTH')

else:
    if page == 'Health Prediction':

        st.title('Mental Health Prediction')    
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCenU9wl-BzzDy_T7C1qotiJlc9NY4r6Vle5HgLlvgKNdu5nJNqK3tViZLH5XJUXeN288&usqp=CAU')
        
        #filters
        age = st.number_input('Enter your Age',value = 0.0)
        working_student = st.selectbox('Select your status',['Working Professional','Student'])
        profession = st.selectbox('Select your profession',valid_professions)
        sleep = st.selectbox('Select your sleep duration',valid_sleep_duration)
        degree = st.selectbox('Select your degree',valid_degree)
        suicidal_thought = st.selectbox('Have you ever had suicidal thoughts ?',['Yes','No'])
        work_study = st.selectbox('Enter your study or working hours',work_study_hrs)
        selected_verbal = st.selectbox("Select Your Financial Stress Level:", list(verbal_to_numeric.keys()))
        selected_numeric = verbal_to_numeric[selected_verbal]
       
        # Preprocessing
        age_scaling = age_scaler.transform([[age]])[0][0]
        working_student_binary_encode = w_s_encoder.transform(pd.DataFrame([{"Working Professional or Student": working_student}])).values[0]
        profession_encoding = pro_encoder.transform(pd.DataFrame([{"Profession": profession}])).values[0]
        sleep_encoding = le_sleep.transform([sleep])[0]
        degree_encode = deg_encoder.transform(pd.DataFrame([{"Degree": degree}])).values[0]
        suicidal_thought_encode = thoughts_encoder.transform(pd.DataFrame([{"Have you ever had suicidal thoughts ?": suicidal_thought}])).values[0]
        work_study_encode = le_work_study.transform([work_study])[0]
        selected_numeric_encode = le_financial.transform([selected_numeric])[0]



        
        input_data = torch.tensor(np.concatenate([[age_scaling], working_student_binary_encode, profession_encoding,
                                              [sleep_encoding], degree_encode, suicidal_thought_encode,
                                              [work_study_encode], [selected_numeric_encode]]),
                              dtype=torch.float32).unsqueeze(0)

        if st.button('Predict'):
            with torch.no_grad():
                predictions = clf(input_data).round()
                result = 'Depression' if predictions.item() == 1 else 'No Depression'
                st.success(f'The person has {result}')





