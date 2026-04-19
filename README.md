# 🌾 KrishiAI – Agriculture AI Chatbot

An AI-powered agriculture assistant that helps farmers **detect crop diseases** and provides **treatment recommendations** using Machine Learning and AI.

---

## 🚀 Features

* 🌿 **Disease Detection (ML)**
  Predicts crop diseases from natural language symptoms using Random Forest

* 🧠 **AI Chatbot (LLM)**
  Answers agriculture-related queries using LangChain + Groq (LLaMA)

* 💊 **Treatment Suggestions**
  Provides fertilizers, pesticides, and preventive measures

* 💬 **Interactive UI**
  Built using Streamlit with a clean dark professional interface

* 🔁 **Conversation Memory**
  Chatbot remembers previous messages for better responses

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Model:** Random Forest + TF-IDF
* **AI Chat:** LangChain + Groq API (LLaMA 3)
* **Libraries:**

  * scikit-learn
  * pandas
  * dotenv
  * langchain

---

## 📸 Demo

*Add screenshots here*

```md
![App Screenshot](screenshots/app.png)
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/rahulvigneshkumaran/agriculture-ai-chatbot.git
cd agriculture-ai-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
streamlit run UI.py
```

---

## 📂 Project Structure

```bash
Agriculture-AI-Chatbot/
│
├── UI.py
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
├── screenshots/
└── README.md
```

---

## 🎯 How It Works

1. User enters crop symptoms
2. ML model predicts disease
3. System fetches treatment
4. Chatbot gives additional advice

---

## 📌 Future Improvements

* 📷 Image-based disease detection (CNN)
* 🌦 Weather API integration
* 📱 Mobile optimization
* 🌍 Multi-language support

---

## 👨‍💻 Author

**Rahul Kumaran**
GitHub: https://github.com/rahulvigneshkumaran

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
