import os
import re
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import google.generativeai as genai

# ==========================================
# 0. GEMINI API CONFIGURATION
# ==========================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
genai.configure(api_key=GEMINI_API_KEY)

def get_ai_lifestyle_advice(prediction_label, age, gender, occupation, sleep_hours, stress_level, anxiety_score, depression_score):
    prompt = f"""
    You are an advanced professional psychological and lifestyle advisory AI. 
    Based on the following user metrics and prediction result, provide personalized, structured, actionable lifestyle advice in English.
    
    - Prediction Result: {prediction_label}
    - Age: {age}, Gender: {gender}, Occupation: {occupation}
    - Sleep Hours: {sleep_hours} hrs/night
    - Stress Level: {stress_level}/10
    - Anxiety Score: {anxiety_score}/10
    - Depression Score: {depression_score}/10
    
    Provide recommendations using clear headings with emojis, concise explanations, and professional wellness tips. Do not mention any commercial AI brand names.
    """
    try:
        ai_model = genai.GenerativeModel("gemini-3.8-flash") # Cập nhật model name chuẩn tránh lỗi không tồn tại
        response = ai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Unable to dynamic AI advice. (Error: {e})"

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="MindSync | Mental Health AI",
    page_icon="🌿",
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
# 2. LOAD MODELS (FIXED PATH FOR CLOUD)
# ==========================================
@st.cache_resource
def load_model():
    # Sử dụng đường dẫn tương đối để chạy mượt mà cả trên máy lẫn trên Streamlit Cloud
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_mental_health_model.pkl')
    le_path = os.path.join(current_dir, 'label_encoder.pkl')
    
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le

try:
    model, le = load_model()
except Exception as e:
    st.error(f"❌ Could not load model files! Error details: {e}")
    model, le = None, None

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2877/2877012.png", width=100)
    st.title("MindSync AI 🌿")
    st.markdown("### About the System")
    st.info("AI-powered diagnostic tool for lifestyle analysis and psychological well-being tracking.")
    st.markdown("---")
    st.markdown("🔒 **Data Privacy:** Confidential medical records.")
    st.markdown("👨‍⚕️ **Disclaimer:** For clinical support reference only.")

# ==========================================
# 4. MAIN HERO SECTION
# ==========================================
st.markdown('<p class="hero-title">Mental Health AI Screener</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Empowering your mental well-being through data-driven insights and longitudinal tracking.</p>', unsafe_allow_html=True)

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
if model is not None:
  with st.container():
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
      st.markdown("### 👤 Demographics")
      age = st.number_input("Age", min_value=10, max_value=100, value=25)
      gender = st.selectbox("Gender", ["Male", "Female", "Other"])
      occupation = st.selectbox(
          "Occupation",
          ["Student", "Corporate", "Business", "Unemployed", "Others"],
      )
      academic_work_pressure = st.slider(
          "Work/Academic Pressure (1-10)", 1.0, 10.0, 5.0
      )
      work_life_balance = st.slider(
          "Work-Life Balance Score (1-10)", 1.0, 10.0, 6.0
      )

    with col2:
      st.markdown("### 😴 Lifestyle Metrics")
      sleep_hours = st.slider("Sleep Hours / Night", 2.0, 12.0, 6.5, 0.5)
      sleep_quality = st.slider("Sleep Quality (1-10)", 1.0, 10.0, 7.0)
      social_media_hours = st.slider(
          "Screen Time / Day (Hours)", 0.0, 15.0, 3.5, 0.5
      )
      physical_activity_days = st.slider("Active Days / Week", 0.0, 7.0, 3.0)

    with col3:
      st.markdown("### 📊 Psychological State")
      stress_level = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0)
      mood_score = st.slider("Overall Mood Score (1-10)", 1.0, 10.0, 7.0)
      anxiety_score = st.slider("Anxiety Indicators (1-10)", 1.0, 10.0, 4.0)
      depression_score = st.slider(
          "Depression Indicators (1-10)", 1.0, 10.0, 3.0
      )
      concentration_level = st.slider(
          "Focus & Concentration (1-10)", 1.0, 10.0, 7.0
      )
      social_support = st.slider("Social Support Level (1-10)", 1.0, 10.0, 6.0)

  st.markdown("<br>", unsafe_allow_html=True)

  # ==========================================
  # 7. EXECUTE PREDICTION & VALIDATION
  # ==========================================
  _, center_col, _ = st.columns([1, 2, 1])

  with center_col:
    predict_button = st.button("✨ ANALYZE MY PROFILE ✨")

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
      with st.spinner("AI is analyzing..."):
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
          prediction_num = model.predict(input_df)[0]
          prediction_label = le.inverse_transform([prediction_num])[0]
          probs = model.predict_proba(input_df)[0]

          st.markdown("---")
          st.markdown("## 📋 Comprehensive AI Report")

          res_col1, res_col2 = st.columns([1.5, 1])

          with res_col1:
            if prediction_label.lower() == "normal":
              st.success(f"### 🌿 Condition: {prediction_label.upper()}")
              st.write("Your psychological profile is healthy. Maintain your current work-life balance!")
              st.balloons()
            elif "anxiety" in prediction_label.lower():
              st.warning(f"### ⚠️ Condition: {prediction_label.upper()}")
              st.write("AI detected patterns consistent with elevated anxiety. Professional guidance is advised.")
            elif "depression" in prediction_label.lower():
              st.error(f"### 🛑 Condition: {prediction_label.upper()}")
              st.write("AI detected potential depression indicators. Clinical support is strongly recommended.")
            else:
              st.error(f"### 🛑 Condition: {prediction_label.upper()}")
              st.write("Significant risk factors identified. Clinical consultation recommended.")

            st.markdown("#### 💡 Actionable Lifestyle Advice")
            
            # Gọi Gemini API tự động tạo lời khuyên chuyên sâu
            dynamic_advice = get_ai_lifestyle_advice(
                prediction_label, age, gender, occupation, 
                sleep_hours, stress_level, anxiety_score, depression_score
            )
            st.markdown(dynamic_advice)

          with res_col2:
            st.markdown("#### 🎯 AI Confidence Score")
            for idx, class_name in enumerate(le.classes_):
              st.write(f"**{class_name}:** {probs[idx]*100:.1f}%")
              st.progress(float(probs[idx]))

          # ==========================================
          # 8. DATABASE INTEGRATION (AZURE C# BACKEND)
          # ==========================================
          # Đổi từ localhost sang đường dẫn Azure App Service của bạn
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
            response = requests.post(api_endpoint, json=payload, timeout=5)
            if response.status_code == 200:
              st.success("✅ Assessment saved successfully as a new patient record (Longitudinal tracking updated)!")
            else:
              st.warning(f"⚠️ Server rejected request (Status Code: {response.status_code})")
          except requests.exceptions.RequestException:
            st.caption("ℹ️ *Note: Backend offline or network block, record stored locally only.*")

        except Exception as e:
          st.error(f"Error executing prediction: {e}")
