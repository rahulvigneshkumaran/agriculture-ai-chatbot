import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Agriculture AI Chatbot",
    page_icon="🌾",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7f2; }
    .stApp { background-color: #f5f7f2; }

    .header-box {
        background: white;
        border: 1px solid #e0e8d8;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .stat-box {
        background: #f0f5ea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-label {
        font-size: 12px;
        color: #666;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 22px;
        font-weight: 600;
        color: #3B6D11;
    }
    .ml-badge {
        background: #EAF3DE;
        border: 1px solid #C0DD97;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 13px;
        color: #3B6D11;
        margin-bottom: 8px;
        display: inline-block;
        font-weight: 500;
    }
    .tip-box {
        background: #FAEEDA;
        border: 1px solid #FAC775;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #633806;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background: white !important;
        border-radius: 12px !important;
        border: 1px solid #e0e8d8 !important;
        margin-bottom: 8px !important;
    }
    .stChatInputContainer {
        background: white !important;
        border: 1px solid #e0e8d8 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load API Key ──────────────────────────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ── Initialize everything once ────────────────────────
@st.cache_resource
def initialize():
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
    )

    data = {
        "yellow_leaves": [1,1,0,0,1,0,1,0,0,1,1,0,0,1,0],
        "brown_spots":   [0,1,1,0,0,1,0,1,0,0,1,1,0,0,1],
        "wilting":       [0,0,0,1,1,1,0,0,1,0,0,0,1,1,1],
        "white_powder":  [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
        "leaf_curling":  [1,0,0,0,1,0,0,0,0,1,0,0,0,1,0],
        "disease": [
            "Nitrogen Deficiency","Early Blight","Early Blight",
            "Root Rot","Nitrogen Deficiency","Early Blight",
            "Powdery Mildew","Powdery Mildew","Root Rot",
            "Nitrogen Deficiency","Early Blight","Early Blight",
            "Root Rot","Nitrogen Deficiency","Root Rot"
        ]
    }
    df = pd.DataFrame(data)
    X = df[["yellow_leaves","brown_spots","wilting","white_powder","leaf_curling"]]
    y = df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, ml_model.predict(X_test)) * 100

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert agriculture assistant
         ONLY helping  farmers with:
         - Crop diseases and treatments
         - Fertilizers and pesticides
         - Farming techniques
         - Soil and water management
         - Weather based farming advice

         IMPORTANT RULES:
         - ONLY answer agriculture related questions
         - If asked anything unrelated, politely say:
           "I can only help with agriculture related questions!"
         - Always respond in simple English
         """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return ml_model, chatbot, round(accuracy, 1)

ml_model, chatbot, accuracy = initialize()

# ── ML Prediction ─────────────────────────────────────
def predict_disease_ml(user_input: str):
    symptom_keywords = {
        "yellow_leaves": ["yellow", "yellowing", "pale"],
        "brown_spots":   ["brown", "spots", "patches"],
        "wilting":       ["wilt", "drooping", "dying"],
        "white_powder":  ["white", "powder", "mildew"],
        "leaf_curling":  ["curl", "curling", "twisted"]
    }
    symptoms = {}
    user_lower = user_input.lower()
    for symptom, keywords in symptom_keywords.items():
        symptoms[symptom] = 1 if any(
            k in user_lower for k in keywords
        ) else 0
    if sum(symptoms.values()) > 0:
        input_data = pd.DataFrame([symptoms])
        disease = ml_model.predict(input_data)[0]
        confidence = ml_model.predict_proba(input_data).max() * 100
        return disease, round(float(confidence), 1), symptoms
    return None, None, symptoms

# ── Safe Invoke ───────────────────────────────────────
def safe_invoke(question: str):
    config = {"configurable": {"session_id": "farmer_1"}}
    try:
        return chatbot.invoke({"question": question}, config=config)
    except Exception as e:
        if "429" in str(e):
            time.sleep(30)
            try:
                return chatbot.invoke({"question": question}, config=config)
            except:
                return "Rate limit hit. Please wait 1 minute and try again."
        return f"Error: {e}"

# ── Header ────────────────────────────────────────────
st.markdown("""
<div style="background:white;border:1px solid #e0e8d8;border-radius:12px;
padding:1.5rem 2rem;margin-bottom:1rem;">
<h2 style="color:#27500A;margin:0;font-size:22px">🌾 Agriculture AI Chatbot</h2>
<p style="color:#666;margin:4px 0 0;font-size:14px">
AI powered assistant for Indian farmers</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="stat-box">
    <div class="stat-label">ML Model</div>
    <div class="stat-value">Ready</div></div>""",
    unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="stat-box">
    <div class="stat-label">Diseases known</div>
    <div class="stat-value">5</div></div>""",
    unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-box">
    <div class="stat-label">Accuracy</div>
    <div class="stat-value">{accuracy}%</div></div>""",
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tip Box ───────────────────────────────────────────
st.markdown("""
<div class="tip-box">
Tip: describe symptoms like "yellow leaves", "brown spots",
"wilting" for best ML detection
</div>
""", unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Vanakkam! I am your Agriculture AI assistant. Describe your crop symptoms and I will diagnose the disease and suggest treatment!",
        "badge": None
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("badge"):
            st.markdown(
                f'<div class="ml-badge">{msg["badge"]}</div>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────
if user_input := st.chat_input("Describe your crop symptoms..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "badge": None
    })

    disease, confidence, symptoms = predict_disease_ml(user_input)
    detected = [k for k, v in symptoms.items() if v == 1]

    if disease:
        question = f"""
        Farmer says: {user_input}
        ML detected: {disease} ({confidence}% confidence)
        Symptoms: {detected}
        Give treatment steps, prevention tips, fertilizer recommendations.
        """
        badge = f"ML detected: {disease} — {confidence}% confidence"
    else:
        question = user_input
        badge = None

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = safe_invoke(question)
        if badge:
            st.markdown(
                f'<div class="ml-badge">{badge}</div>',
                unsafe_allow_html=True
            )
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "badge": badge
    })