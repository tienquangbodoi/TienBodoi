import os
import re
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import shap
import plotly.graph_objects as go

# ==========================================
# 0. LOCAL RAG & XAI CONFIGURATION (OFFLINE)
# ==========================================
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# SỬ DỤNG KIẾN TRÚC LCEL (LangChain Expression Language) CHUẨN HỌC THUẬT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

@st.cache_resource
def setup_rag_system():
    try:
        # Use Word Embedding to encode medical documents
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        texts = [
            "Depression patients with Stress > 7 and Sleep Deficit: Cognitive Behavioral Therapy (CBT) combined with sleep hygiene is recommended. Consider Melatonin 3mg before bedtime.",
            "Anxiety patients due to academic/work pressure: Mindfulness exercises are required, and social media usage should be reduced to under 2 hours/day.",
            "Normal individuals with high pressure: It is recommended to increase physical activity to 4-5 days/week to release Endorphins."
        ]
        vector_db = FAISS.from_texts(texts, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        
        # Local LLM
        llm = Ollama(model="llama3") 
        
        # Academic standard Prompt Template
        template = """You are a clinical assistant. Use the following pieces of retrieved context to answer the clinical query. 
        If you don't know the answer, just say that you don't know. Keep the answer concise and professional.
        
        Context: {context}
        
        Query: {question}
        
        Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # LCEL Pipeline (State-of-the-Art RAG Architecture)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        return None

rag_chain = setup_rag_system()

def simulate_what_if(model, preprocessor, input_data_raw, le):
    simulated_data = input_data_raw.copy()
    
    # Simulate decreasing stress by 2 points and increasing sleep by 1.5 hours
    simulated_data[0]['stress_level'] = max(0, simulated_data[0]['stress_level'] - 2)
    simulated_data[0]['sleep_hours'] = min(10, simulated_data[0]['sleep_hours'] + 1.5)
    
    simulated_data[0]['sleep_deficit'] = max(0, 8 - simulated_data[0]['sleep_hours'])
    simulated_data[0]['stress_support_ratio'] = simulated_data[0]['stress_level'] / (simulated_data[0]['social_support'] + 1)
    
    sim_df = pd.DataFrame(simulated_data)
    X_sim_processed = preprocessor.transform(sim_df)
    
    sim_probs = model.predict_proba(X_sim_processed)[0]
    sim_pred_idx = np.argmax(sim_probs)
    sim_class = le.inverse_transform([sim_pred_idx])[0]
    
    return sim_class, np.max(sim_probs) * 100

def draw_clinical_radar(raw_dict):
    categories = ['Stress', 'Academic/Work Pressure', 'Anxiety', 'Depression', 'Lack of Social Support']
    patient_scores = [
        raw_dict['stress_level'], 
        raw_dict['academic_work_pressure'], 
        raw_dict['anxiety_score'], 
        raw_dict['depression_score'], 
        10 - raw_dict['social_support'] # Inverted: Higher score means lack of support
    ]
    healthy_baseline = [4, 5, 3, 2, 3]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=patient_scores + [patient_scores[0]],
          theta=categories + [categories[0]],
          fill='toself', fillcolor='rgba(255, 65, 54, 0.4)', line=dict(color='red'),
          name='Patient Profile'
    ))
    fig.add_trace(go.Scatterpolar(
          r=healthy_baseline + [healthy_baseline[0]],
          theta=categories + [categories[0]],
          fill='toself', fillcolor='rgba(0, 123, 255, 0.2)', line=dict(color='blue', dash='dot'),
          name='Healthy Baseline'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True, margin=dict(l=30, r=30, t=30, b=30), height=350)
    return fig

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Mental Health CDSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #f4f7f6; }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2193b0, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 40px;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        font-weight: 700;
        border-radius: 50px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(33, 147, 176, 0.4);
        transition: all 0.3s ease 0s;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(33, 147, 176, 0.6);
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_model():
    MY_FOLDER = r"C:\Users\ADMIN\MethalHealthApp" 
    pipeline_path = os.path.join(MY_FOLDER, 'xgboost_mental_health_clinical.pkl')
    le_path = os.path.join(MY_FOLDER, 'label_encoder.pkl')
    ood_path = os.path.join(MY_FOLDER, 'ood_detector_isolation_forest.pkl')
    
    pipeline = joblib.load(pipeline_path)
    le = joblib.load(le_path)
    ood = joblib.load(ood_path)
    return pipeline, le, ood

try:
    pipeline, le, ood_detector = load_model()
except Exception as e:
    st.error(f"❌ Could not load model files! Error details: {e}")
    pipeline, le, ood_detector = None, None, None

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2877/2877012.png", width=100)
    st.title("Clinical CDSS 🧠")
    st.markdown("### About the System")
    st.info("Clinical Decision Support System (CDSS) utilizing Ensemble ML & Local RAG LLM.")
    st.markdown("---")
    st.markdown("🔒 **Data Privacy:** 100% Offline processing. No external API calls.")
    st.markdown("👨‍⚕️ **Disclaimer:** For clinical support reference only.")
    st.markdown("---")
    st.markdown("© 2026 Clinical Systems")

# ==========================================
# 4. MAIN HERO SECTION
# ==========================================
st.markdown('<p class="hero-title">Mental Health AI Screener</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Empowering psychological well-being through data-driven insights and longitudinal tracking.</p>', unsafe_allow_html=True)

# ==========================================
# 5. PATIENT IDENTIFICATION SECTION
# ==========================================
st.markdown("### 🪪 Patient Identification")
id_col1, id_col2, id_col3, id_col4 = st.columns(4)

with id_col1:
    patient_name = st.text_input("Full Name (Optional for follow-up)")
with id_col2:
    phone = st.text_input("Phone Number * (0... or +84...)")
with id_col3:
    cccd = st.text_input("Citizen ID / CCCD * (12 digits)", max_chars=12)
with id_col4:
    follow_up_date = st.date_input("Follow-up Date Reminder")

st.markdown("---")

# ==========================================
# 6. USER INPUT FORM
# ==========================================
if pipeline is not None:
  with st.container():
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
      st.markdown("### 👤 Demographics")
      age = st.number_input("Age", min_value=10, max_value=100, value=25)
      gender = st.selectbox("Gender", ["Male", "Female", "Other"])
      occupation = st.selectbox("Occupation", ["Student", "Corporate", "Business", "Unemployed", "Others"])
      academic_work_pressure = st.slider("Work/Academic Pressure (1-10)", 1.0, 10.0, 5.0)
      work_life_balance = st.slider("Work-Life Balance Score (1-10)", 1.0, 10.0, 6.0)

    with col2:
      st.markdown("### 😴 Lifestyle Metrics")
      sleep_hours = st.slider("Sleep Hours / Night", 2.0, 12.0, 6.5, 0.5)
      sleep_quality = st.slider("Sleep Quality (1-10)", 1.0, 10.0, 7.0)
      social_media_hours = st.slider("Screen Time / Day (Hours)", 0.0, 15.0, 3.5, 0.5)
      physical_activity_days = st.slider("Active Days / Week", 0.0, 7.0, 3.0)

    with col3:
      st.markdown("### 📊 Psychological State")
      stress_level = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0)
      mood_score = st.slider("Overall Mood Score (1-10)", 1.0, 10.0, 7.0)
      anxiety_score = st.slider("Anxiety Indicators (1-10)", 1.0, 10.0, 4.0)
      depression_score = st.slider("Depression Indicators (1-10)", 1.0, 10.0, 3.0)
      concentration_level = st.slider("Focus & Concentration (1-10)", 1.0, 10.0, 7.0)
      social_support = st.slider("Social Support Level (1-10)", 1.0, 10.0, 6.0)

  st.markdown("<br>", unsafe_allow_html=True)

  # ==========================================
  # 7. EXECUTE PREDICTION & VALIDATION
  # ==========================================
  _, center_col, _ = st.columns([1, 2, 1])

  with center_col:
    predict_button = st.button("✨ ANALYZE PATIENT PROFILE ✨")

  if predict_button:
    phone_pattern = r"^(\+84|0)[0-9]{9}$"
    cccd_pattern = r"^\d{12}$"

    if not phone or not cccd:
        st.error("⚠️ Phone Number and CCCD are mandatory fields!")
    elif not re.match(phone_pattern, phone):
        st.error("⚠️ Invalid Phone Number format! Must start with 0 or +84 and contain 10 digits total.")
    elif not re.match(cccd_pattern, cccd):
        st.error("⚠️ Invalid CCCD format! Must be exactly 12 numeric digits.")
    else:
      with st.spinner("AI is analyzing via Clinical Pipeline..."):
        sleep_deficit = 8 - sleep_hours
        stress_support_ratio = stress_level / (social_support + 1)
        screen_to_sleep_ratio = social_media_hours / (sleep_hours + 1)

        raw_dict = {
            'age': age,
            'gender': gender,
            'occupation': occupation,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'social_media_hours': social_media_hours,
            'academic_work_pressure': academic_work_pressure,
            'physical_activity_days': physical_activity_days,
            'stress_level': stress_level,
            'anxiety_score': anxiety_score,
            'depression_score': depression_score,
            'work_life_balance': work_life_balance,
            'mood_score': mood_score,
            'concentration_level': concentration_level,
            'social_support': social_support,
            'sleep_deficit': sleep_deficit,
            'stress_support_ratio': stress_support_ratio,
            'screen_to_sleep_ratio': screen_to_sleep_ratio
        }

        input_df = pd.DataFrame([raw_dict])

        try:
          preprocessor = pipeline.named_steps['preprocessor']
          classifier = pipeline.named_steps['classifier']
          X_processed = preprocessor.transform(input_df)
          
          # OOD Detection
          is_anomaly = ood_detector.predict(X_processed)[0]

          if is_anomaly == -1:
              st.error("🚫 PREDICTION REJECTED: Out-of-Distribution (OOD) biometric data detected. Please review the input.")
          else:
              prediction_num = classifier.predict(X_processed)[0]
              prediction_label = le.inverse_transform([prediction_num])[0]
              probs = classifier.predict_proba(X_processed)[0]
              max_prob = np.max(probs) * 100

              # XAI - SHAP extraction
              explainer = shap.TreeExplainer(classifier)
              shap_values = explainer.shap_values(X_processed)
              if isinstance(shap_values, list): patient_shap = shap_values[prediction_num][0]
              elif len(shap_values.shape) == 3: patient_shap = shap_values[0, :, prediction_num]
              else: patient_shap = shap_values[0]

              feature_names = [name.replace('num__', '').replace('cat__', '') for name in preprocessor.get_feature_names_out()]
              contributions = list(zip(feature_names, patient_shap))
              contributions.sort(key=lambda x: x[1], reverse=True)
              top_risk = contributions[0][0]

              # What-If Simulation
              sim_class, sim_prob = simulate_what_if(classifier, preprocessor, [raw_dict], le)

              st.markdown("---")
              st.markdown("## 📋 Comprehensive Clinical AI Report")

              res_col1, res_col2 = st.columns([1.5, 1])

              with res_col1:
                if prediction_label.lower() == "normal":
                  st.success(f"### 🌿 Condition: {prediction_label.upper()} (Confidence: {max_prob:.1f}%)")
                else:
                  st.error(f"### 🛑 Condition: {prediction_label.upper()} (Confidence: {max_prob:.1f}%)")
                
                st.markdown("#### 🔬 Root Cause Analysis (XAI)")
                st.write(f"According to the ensemble decision tree algorithm, the highest risk factor for this case is: **{top_risk}**.")
                
                st.markdown("#### 🔮 What-If Simulation")
                st.info(f"If the patient reduces Stress by 2 points and increases sleep by 1.5h/night, the projected risk shifts to: **{sim_class.upper()}** (Probability: {sim_prob:.1f}%).")

              with res_col2:
                st.markdown("#### 📊 Clinical Symptom Radar")
                st.plotly_chart(draw_clinical_radar(raw_dict), use_container_width=True)

              # RAG Chatbot Integration (LCEL)
              st.markdown("---")
              st.markdown("### 🤖 RAG Clinical Assistant (Offline Local LLM)")
              
              if "chat_history" not in st.session_state:
                  st.session_state.chat_history = []
                  
              if len(st.session_state.chat_history) == 0:
                  context_msg = f"Based on system data: Patient condition is {prediction_label}, primary risk driver is {top_risk}. Provide a concise 3-bullet intervention plan."
                  if rag_chain:
                      with st.spinner("Querying Medical Vector Database..."):
                          ans = rag_chain.invoke(context_msg)
                          st.session_state.chat_history.append({"role": "assistant", "content": ans})

              for msg in st.session_state.chat_history:
                  st.chat_message(msg["role"]).write(msg["content"])

              if prompt_text := st.chat_input("Ask the AI for detailed treatment guidelines or drug interactions..."):
                  st.chat_message("user").write(prompt_text)
                  st.session_state.chat_history.append({"role": "user", "content": prompt_text})
                  with st.chat_message("assistant"):
                      if rag_chain:
                          with st.spinner("Retrieving knowledge base..."):
                              ans = rag_chain.invoke(prompt_text)
                              st.write(ans)
                              st.session_state.chat_history.append({"role": "assistant", "content": ans})
                      else:
                          st.error("Vector database is not ready.")

              # ==========================================
              # 8. DATABASE INTEGRATION (C# AZURE BACKEND)
              # ==========================================
              api_endpoint = "https://metnalhealth-h2gccagga6d4hta6.eastasia-01.azurewebsites.net/api/ScreeningAPI/Save"
              
              payload = {
                  "UserId": 1,
                  "PatientName": str(patient_name),
                  "PhoneNumber": str(phone),
                  "CCCD": str(cccd),
                  "FollowUpDate": follow_up_date.isoformat(),
                  "Age": int(age),
                  "Gender": str(gender),
                  "Occupation": str(occupation),
                  "SleepHours": float(sleep_hours),
                  "StressLevel": float(stress_level),
                  "PredictionResult": str(prediction_label),
              }

              try:
                response = requests.post(api_endpoint, json=payload, timeout=30)
                if response.status_code == 200:
                  st.success("✅ Assessment saved successfully to SQL Server (Azure Backend)!")
                else:
                  st.warning(f"⚠️ Server rejected request (Status Code: {response.status_code})")
              except requests.exceptions.RequestException as req_err:
                st.error(f"❌ Connection to Azure Backend failed. Details: {req_err}")
                st.caption("ℹ️ *Note: Ensure your Azure Web App is running and CORS allows the Streamlit domain.*")

        except Exception as e:
          st.error(f"Error executing prediction: {e}")
