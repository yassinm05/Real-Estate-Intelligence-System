# 🏡 Seattle Real Estate AI Agent 
**A Hybrid Local/Cloud RAG & Machine Learning Ecosystem**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=flat&logo=google&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172A3C?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=flat)

This repository contains a comprehensive, hybrid AI ecosystem operating as an intelligent Seattle Real Estate Agent. By combining a highly optimized **Retrieval-Augmented Generation (RAG)** pipeline with advanced **Machine Learning (ML)** and **Deep Learning (DL)** models, the system goes beyond simple chat functionality. It intelligently queries over 84,000+ real estate reviews, runs sentiment analysis, and predicts fair-market property prices in real-time.

---

## ✨ Features

* **Intelligent QA & Recommendation (RAG):** Answers complex, hyper-specific user queries by retrieving context from over 84,000 embedded property reviews using LangChain and Google Gemini 2.5 Flash.
* **Fair-Market Price Prediction (ML):** Integrates an optimized XGBoost predictive model achieving a highly accurate **$30 MSE**, allowing the system to evaluate real estate pricing dynamics. 
* **Custom Deep Learning Embeddings:** Utilizes a fine-tuned DistilBERT model trained on Seattle Airbnb data. Implemented entirely locally via pure PyTorch and Safetensors with custom Mean Pooling logic for highly accurate semantic representation.
* **Microservices Architecture:** Built on a robust, asynchronous FastAPI backend paired with an intuitive, interactive Streamlit chat interface.
* **Local Vector Memory:** Leverages a persistent local ChromaDB instance to guarantee low-latency vector similarity search without relying on external vector cloud providers.

---

## 🏗️ System Architecture & Tech Stack

The platform is divided into distinct, purpose-built components:

* **The Backend Orchestrator:** **FastAPI** running on Uvicorn, serving as the central nervous system connecting the frontend to the vector database and ML/DL models.
* **The Frontend UI:** **Streamlit**, providing a clean, responsive, and interactive chat interface for users.
* **The Brain (LLM):** **Google Gemini 2.5 Flash** (interfaced via `langchain-google-genai`) for synthesizing retrieved context into professional, grounded advice.
* **The Embedder (DL):** A **DistilBERT** model fine-tuned for real estate sentiment and semantics, mapping review text into 768-dimensional vectors.
* **The Predictor (ML):** **XGBoost**, utilizing a joblib-serialized preprocessing pipeline (`scaler.pkl`, `encoder.pkl`) and a natively saved JSON model (`xgboost_price_model.json`).
* **The Memory (Vector DB):** **ChromaDB**, functioning as the local, persistent vector store.
* **Data Engineering:** **Pandas** and **Scikit-Learn** handle data cleaning, feature engineering, and pipeline execution.

---

## 🔄 System Flow

1. **Ingestion & Training:** Raw CSVs of listings and reviews are cleaned. The XGBoost model learns property features to predict prices. DistilBERT embeds text into 768-dimensional vectors. Both vectors and crucial metadata are persisted into ChromaDB.
2. **Retrieval:** A user enters a prompt into the Streamlit UI. The request is routed to FastAPI. DistilBERT embeds the user's query, and ChromaDB performs a Cosine Similarity search to retrieve the top 5 most relevant property profiles.
3. **Generation:** LangChain injects the retrieved context and metadata into a strict system prompt. Gemini evaluates the prompt and generates a grounded, highly professional response. *(Note: Future architecture updates will automatically inject the XGBoost price prediction into this context window to flag over/under-priced properties).*

---

## 📂 Repository Structure

```plaintext
real_estate_rag_ecosystem/
├── data/                 # Raw and preprocessed CSV datasets
├── models/               
│   ├── my_real_estate_distilbert/  # Fine-tuned safetensors and configs
│   ├── xgboost_price_model.json    # Trained ML price predictor
│   ├── scaler.pkl                  # Joblib serialized feature scaler
│   └── encoder.pkl                 # Joblib serialized categorical encoder
├── notebooks/            # Jupyter notebooks for EDA, ML/DL Training, and Ingestion
├── chroma_db/            # Local vector database storage
├── app/                  
│   ├── main.py           # FastAPI server and endpoints
│   └── rag_engine.py     # Core RAG logic, LangChain, and ChromaDB connection
├── frontend.py           # Streamlit chat interface
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (Google API Key)
```

---

## ⚙️ Environment Variables

To run the application, you must configure your environment variables. Create a `.env` file in the root directory of the project and add your Google API key.
```bash
# .env file
GOOGLE_API_KEY="your_google_gemini_api_key_here"
```

---

## 🚀 Local Setup & Installation

Follow these steps to get the hybrid ecosystem running on your local machine. Because this is a decoupled architecture, **you must run the backend and frontend simultaneously in separate terminal windows.**

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/yourusername/real_estate_rag_ecosystem.git](https://github.com/yourusername/real_estate_rag_ecosystem.git)
cd real_estate_rag_ecosystem
```

**2. Create and activate a virtual environment:**
```bash
# For MacOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Start the Backend (Terminal 1):**
Keep this terminal open. This boots up the FastAPI orchestrator, initializes ChromaDB, and loads the deep learning models into memory.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**5. Start the Frontend (Terminal 2):**
Open a new terminal, activate your virtual environment again, and launch the Streamlit chat interface.
```bash
streamlit run frontend.py
```

Once both services are running, Streamlit will automatically open a browser window (usually at `http://localhost:8501`) where you can interact with the Seattle Real Estate AI Agent.
