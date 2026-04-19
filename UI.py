import os
import time
import uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(
    page_title="KrishiAI — Smart Farming Assistant",
    page_icon="🌾",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, .stApp {
    background-color: #0a1628 !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 0 !important;
    max-width: 780px;
    position: relative;
    z-index: 1;
}

.nav {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
}
.nav-logo { display: flex; align-items: center; gap: 10px; }
.nav-icon {
    width: 38px; height: 38px; background: #52b788;
    border-radius: 10px; display: flex;
    align-items: center; justify-content: center; font-size: 20px;
}
.nav-name { font-size: 20px; font-weight: 800; color: #ffffff; }
.nav-name span { color: #52b788; }
.nav-tagline { font-size: 12px; color: rgba(255,255,255,0.55); }

.hero { text-align: center; padding: 2rem 0 1.5rem; }
.hero-badge {
    display: inline-block;
    background: rgba(82,183,136,0.15);
    border: 1px solid rgba(82,183,136,0.3);
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; color: #52b788; margin-bottom: 1rem;
}
.hero h1 { font-size: 38px; font-weight: 800; color: #ffffff; line-height: 1.2; margin-bottom: 1rem; }
.hero h1 span { color: #52b788; }
.hero p { font-size: 15px; color: rgba(255,255,255,0.72); max-width: 480px; margin: 0 auto 2rem; line-height: 1.7; }

.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2rem; }
.stat-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(8px);
}
.stat-num { font-size: 22px; font-weight: 700; color: #52b788; }
.stat-lbl { font-size: 11px; color: rgba(255,255,255,0.65); margin-top: 3px; }

.features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 2rem; }
.feat-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px; padding: 1.2rem; backdrop-filter: blur(8px);
}
.feat-icon { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 10px; }
.feat-icon.green { background: rgba(82,183,136,0.2); }
.feat-icon.blue  { background: rgba(55,138,221,0.2); }
.feat-icon.amber { background: rgba(239,159,39,0.2); }
.feat-title { font-size: 13px; font-weight: 600; color: #ffffff; margin-bottom: 5px; }
.feat-desc { font-size: 12px; color: rgba(255,255,255,0.70); line-height: 1.6; }

.chat-header-box {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px 14px 0 0; padding: 1rem 1.2rem;
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.chat-dot { width: 8px; height: 8px; border-radius: 50%; background: #52b788; }
.chat-title-text { font-size: 14px; font-weight: 600; color: #ffffff; }
.chat-online { margin-left: auto; font-size: 11px; color: #52b788; }

.ml-badge {
    display: inline-block; background: rgba(82,183,136,0.15);
    border: 1px solid rgba(82,183,136,0.3); border-radius: 8px;
    padding: 4px 10px; font-size: 12px; color: #52b788;
    margin-bottom: 6px; font-weight: 600;
}

.tip-box {
    background: rgba(239,159,39,0.1); border: 1px solid rgba(239,159,39,0.25);
    border-radius: 10px; padding: 10px 14px;
    font-size: 12px; color: #ffd08a; margin-bottom: 1rem;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 16px !important;
    padding: 0.75rem 0.95rem !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(8px);
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(82,183,136,0.16) !important;
    border: 1px solid rgba(82,183,136,0.28) !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(55,138,221,0.16) !important;
    border: 1px solid rgba(55,138,221,0.28) !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] label,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em,
div[data-testid="stChatMessage"] div {
    color: #f8fbff !important;
    opacity: 1 !important;
}
div[data-testid="stChatMessage"] ul,
div[data-testid="stChatMessage"] ol { color: #f8fbff !important; }
div[data-testid="stChatMessageContent"] { color: #f8fbff !important; }

/* ── CHAT INPUT FIX — THE KEY PART ───────────────── */
div[data-testid="stChatInput"] {
    background: rgba(14, 24, 40, 0.96) !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    padding-top: 10px !important;
}
div[data-testid="stChatInput"] > div {
    background: rgba(20, 30, 50, 0.98) !important;
    border: 1px solid rgba(82,183,136,0.35) !important;
    border-radius: 16px !important;
}
div[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    /* NO transparent background — this was causing invisible text */
    background: rgba(20, 30, 50, 0.98) !important;
    font-size: 15px !important;
    caret-color: #52b788 !important;
    opacity: 1 !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.45) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.45) !important;
}
div[data-testid="stChatInput"] button {
    background: rgba(82,183,136,0.18) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(82,183,136,0.25) !important;
}

/* Clear button */
.stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important; font-size: 12px !important;
}
.stButton > button:hover {
    background: rgba(255,0,0,0.12) !important;
    color: #ff8a8a !important;
    border-color: rgba(255,0,0,0.22) !important;
}

footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }

@media (max-width: 768px) {
    .stats-row { grid-template-columns: repeat(2, 1fr); }
    .features { grid-template-columns: 1fr; }
    .hero h1 { font-size: 30px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp::before {
    content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(82,183,136,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(82,183,136,0.04) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

@st.cache_resource
def initialize():
    diseases_data = {
        "disease": [
            "Tomato Early Blight","Tomato Late Blight","Tomato Leaf Mold","Tomato Mosaic Virus",
            "Tomato Yellow Leaf Curl","Tomato Septoria Spot","Potato Early Blight","Potato Late Blight",
            "Potato Black Scurf","Potato Common Scab","Rice Blast","Rice Brown Spot",
            "Rice Bacterial Blight","Rice Sheath Blight","Wheat Rust","Wheat Powdery Mildew",
            "Wheat Septoria","Wheat Fusarium","Cotton Boll Rot","Cotton Leaf Curl Virus",
            "Maize Gray Leaf Spot","Maize Common Rust","Maize Northern Blight",
            "Nitrogen Deficiency","Phosphorus Deficiency","Potassium Deficiency","Iron Deficiency",
            "Root Rot","Powdery Mildew","Downy Mildew"
        ],
        "symptoms": [
            "brown spots on leaves yellow halo around spots wilting",
            "water soaked lesions brown patches white mold on leaves",
            "yellow spots on upper leaves brown mold below leaf curl",
            "mosaic pattern on leaves distorted growth stunted plant",
            "yellowing leaves curling upward stunted growth",
            "small circular spots dark brown center yellow margin",
            "brown spots with yellow border leaf wilting stem lesions",
            "water soaked spots white mold rapid wilting brown tubers",
            "black patches on stem base brown tubers stunted growth",
            "rough patches on tuber corky lesions crusty skin",
            "diamond shaped lesions gray center brown border leaf falling",
            "oval brown spots yellow halo lesions on leaves",
            "water soaked lesions yellow edges wilting foul smell",
            "water soaked lesions on sheath brown irregular patches",
            "orange red pustules on leaves yellow stripes wilting",
            "white powdery coating yellow leaves stunted growth",
            "brown blotches on leaves premature aging yield loss",
            "bleached spikelets pink mold shriveled grains",
            "brown rotting bolls mold growth wet conditions",
            "curling leaves mosaic pattern stunted plant thickened veins",
            "long gray streaks on leaves tan lesions leaf blight",
            "orange pustules on leaves yellow circles rust powder",
            "long tan lesions brown edges leaf blight wilting",
            "yellowing lower leaves pale green color stunted growth",
            "purple red discoloration dark green then purple leaves",
            "brown leaf edges scorching weak stems low fruit quality",
            "yellowing between veins green veins visible pale leaves",
            "root decay yellowing leaves wilting foul smell",
            "white powdery coating on leaves stunted growth distortion",
            "yellow patches gray mold on underside leaf distortion"
        ],
        "treatment": [
            "Apply mancozeb fungicide remove infected leaves crop rotation",
            "Apply copper fungicide destroy infected plants avoid overhead irrigation",
            "Apply chlorothalonil improve air circulation remove infected leaves",
            "Remove infected plants control whiteflies use resistant varieties",
            "Control whiteflies use yellow sticky traps resistant varieties",
            "Apply mancozeb remove lower leaves avoid leaf wetness",
            "Apply mancozeb fungicide destroy infected tubers crop rotation",
            "Apply metalaxyl fungicide destroy infected plants immediately",
            "Use certified seed soil solarization avoid waterlogging",
            "Use certified disease free seed soil treatment proper pH",
            "Apply tricyclazole fungicide flood management resistant varieties",
            "Apply mancozeb fungicide balanced fertilization crop rotation",
            "Apply copper bactericide remove infected plants drain fields",
            "Apply carbendazim fungicide avoid dense planting drain water",
            "Apply propiconazole fungicide resistant varieties crop rotation",
            "Apply sulfur fungicide improve air circulation remove infected parts",
            "Apply mancozeb fungicide balanced nutrition crop rotation",
            "Apply fungicide at flowering remove infected heads dry storage",
            "Apply copper fungicide remove infected bolls avoid waterlogging",
            "Control whiteflies use virus free seeds remove infected plants",
            "Apply strobilurin fungicide resistant varieties crop rotation",
            "Apply mancozeb fungicide resistant varieties crop rotation",
            "Apply strobilurin fungicide remove infected leaves crop rotation",
            "Apply urea 2kg per acre compost manure balanced NPK fertilizer",
            "Apply DAP fertilizer bone meal balanced NPK",
            "Apply potash MOP fertilizer wood ash balanced NPK",
            "Apply chelated iron spray lower soil pH foliar iron spray",
            "Improve drainage reduce waterlogging apply fungicide drench",
            "Apply sulfur fungicide neem oil spray improve ventilation",
            "Apply metalaxyl fungicide improve drainage reduce humidity"
        ]
    }
    df = pd.DataFrame(diseases_data)
    X, y = df["symptoms"], df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ml_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    ml_pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, ml_pipeline.predict(X_test)) * 100
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are KrishiAI, an expert agriculture assistant ONLY helping Indian farmers with:
         - Crop diseases and treatments
         - Fertilizers and pesticides
         - Farming techniques
         - Soil and water management
         RULES:
         - ONLY answer agriculture questions
         - If unrelated say: I can only help with farming and agriculture questions!
         - Keep responses clear and practical
         - Always respond in simple English
         - Never repeat the same answer twice
         - Give fresh specific advice each time"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    chatbot = RunnableWithMessageHistory(chain, get_session_history,
        input_messages_key="question", history_messages_key="history")
    return ml_pipeline, chatbot, df, round(accuracy, 1)

ml_pipeline, chatbot, df, accuracy = initialize()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def predict_disease(user_input: str):
    try:
        disease = ml_pipeline.predict([user_input])[0]
        confidence = ml_pipeline.predict_proba([user_input]).max() * 100
        treatment = df[df["disease"] == disease]["treatment"].values[0]
        return disease, round(float(confidence), 1), treatment
    except Exception:
        return None, None, None

def safe_invoke(question: str):
    config = {"configurable": {"session_id": st.session_state.session_id}}
    try:
        return chatbot.invoke({"question": question}, config=config)
    except Exception as e:
        if "429" in str(e):
            time.sleep(15)
            try:
                return chatbot.invoke({"question": question}, config=config)
            except Exception:
                return "Rate limit hit. Please try again shortly."
        return f"Error: {e}"

st.markdown(f"""
<div class="nav">
  <div class="nav-logo">
    <div class="nav-icon">🌾</div>
    <div>
      <div class="nav-name">Krishi<span>AI</span></div>
      <div class="nav-tagline">Smart farming assistant</div>
    </div>
  </div>
  <div style="font-size:12px;color:rgba(255,255,255,0.65)">
    ML: Ready &nbsp;|&nbsp; Diseases: {len(df)} &nbsp;|&nbsp; Accuracy: {accuracy}%
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-badge">Powered by LangChain + Groq AI</div>
  <h1>Smart farming starts<br>with <span>KrishiAI</span></h1>
  <p>AI powered crop disease detection and farming assistant
  built for Indian farmers. Describe your symptoms,
  get instant diagnosis and treatment advice.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
  <div class="stat-card"><div class="stat-num">{len(df)}+</div><div class="stat-lbl">Diseases known</div></div>
  <div class="stat-card"><div class="stat-num">{accuracy}%</div><div class="stat-lbl">ML Accuracy</div></div>
  <div class="stat-card"><div class="stat-num">10+</div><div class="stat-lbl">Crops covered</div></div>
  <div class="stat-card"><div class="stat-num">24/7</div><div class="stat-lbl">Always available</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features">
  <div class="feat-card">
    <div class="feat-icon green">🔬</div>
    <div class="feat-title">ML Disease Detection</div>
    <div class="feat-desc">Random Forest detects diseases from natural language descriptions</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon blue">🧠</div>
    <div class="feat-title">AI Conversation</div>
    <div class="feat-desc">LangChain chatbot remembers context and gives personalized advice</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon amber">🌱</div>
    <div class="feat-title">Treatment Advice</div>
    <div class="feat-desc">Instant fertilizer, pesticide and prevention recommendations</div>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("""
    <div class="chat-header-box">
      <div class="chat-dot"></div>
      <div class="chat-title-text">KrishiAI Assistant</div>
      <div class="chat-online">Online</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.markdown("""
<div class="tip-box">
  Tip: describe symptoms naturally — "yellow leaves",
  "white powder", "brown spots", "wilting", "rust dust"
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Vanakkam! I am KrishiAI, your smart farming assistant. I know 30+ crop diseases! Describe your crop symptoms and I will diagnose and suggest treatment instantly.",
        "badge": None
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("badge"):
            st.markdown(f'<div class="ml-badge">{msg["badge"]}</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])

if user_input := st.chat_input("Describe your crop symptoms..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input, "badge": None})

    disease, confidence, treatment = predict_disease(user_input)

    if disease and confidence >= 40:
        question = f"""
        Farmer says: {user_input}
        ML detected: {disease} ({confidence}% confidence)
        Known treatment: {treatment}
        Give fresh specific practical advice for this case.
        """
        badge = f"ML detected: {disease} — {confidence}% confidence"
    else:
        question = user_input
        badge = None

    with st.chat_message("assistant"):
        with st.spinner("KrishiAI is thinking..."):
            response = safe_invoke(question)
        if badge:
            st.markdown(f'<div class="ml-badge">{badge}</div>', unsafe_allow_html=True)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response, "badge": badge})

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-size:12px;color:rgba(255,255,255,0.32)">
KrishiAI — Built with LangChain, Groq, Streamlit and Random Forest for Indian farmers
</div>
""", unsafe_allow_html=True)