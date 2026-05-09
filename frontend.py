import streamlit as st
import requests
import time

# ==========================================
# 1. Page Configuration & CSS
# ==========================================
st.set_page_config(
    page_title="Seattle AI Real Estate Agent",
    page_icon="🏠",
    layout="centered"
)

# A little custom CSS to make the chat look cleaner
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px;}
    .stChatInputContainer {padding-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. App Header
# ==========================================
st.title("🏠 Seattle AI Real Estate Agent")
st.markdown("Ask me anything about the Seattle Airbnb and rental market. I'll search our database of 84,000+ listings to find your perfect match.")

# The URL where your FastAPI backend is running
API_URL = "http://localhost:8000/api/recommend"

# ==========================================
# 3. Session State (Chat History)
# ==========================================
# This keeps the chat history on the screen even when Streamlit reloads
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Seattle Real Estate AI. What kind of property are you looking for today?"}
    ]

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. Chat Input & API Call
# ==========================================
if prompt := st.chat_input("E.g., Find me a romantic 1-bedroom in Queen Anne under $150..."):
    
    # 1. Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the FastAPI Backend
    with st.chat_message("assistant"):
        # We use a spinner so the user knows the AI is "thinking"
        with st.spinner("Searching the database and analyzing reviews..."):
            
            try:
                # Send the POST request to your FastAPI server
                response = requests.post(
                    API_URL, 
                    json={"query": prompt, "n_results": 5},
                    timeout=120 
                )
                
                if response.status_code == 200:
                    data = response.json()
                    agent_reply = data["agent_response"]
                    raw_context = data["raw_context"]
                    
                    # Display the AI's response
                    st.markdown(agent_reply)
                    
                    # Optional: Add an expander to let users see the raw RAG data!
                    with st.expander("🔍 View Raw Database Context"):
                        st.text(raw_context)
                        
                    # Save the response to state
                    st.session_state.messages.append({"role": "assistant", "content": agent_reply})
                    
                else:
                    error_msg = f"⚠️ Backend Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to the backend. Is your FastAPI server running on port 8000?")