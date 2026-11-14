import streamlit as st
import requests
import sys
import json
import os
import threading
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from main import run_pipeline

st.set_page_config(page_title="AI Dev Studio", layout="wide")
st.title("🛠️ Local AI Dev Studio")

# File to store the persistent chat history
CHAT_FILE = "chat_history.json"

def load_chat():
    """Loads chat history from a JSON file, or sets the default greeting."""
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return [{"role": "assistant", "content": "I am your Lead PM. What are we building today?"}]

def save_chat(history):
    """Saves the current chat history to a JSON file."""
    with open(CHAT_FILE, "w") as f:
        json.dump(history, f, indent=4)

# Custom class to hijack terminal output and send it to the web UI safely
class StreamlitConsole:
    def __init__(self, st_element):
        self.st_element = st_element
        self.logs = ""
        # 1. Capture the main Streamlit thread's context
        self.ctx = get_script_run_ctx() 
        
    def write(self, text):
        # 2. Inject that context into whatever background thread CrewAI is using
        add_script_run_ctx(threading.current_thread(), self.ctx)
        
        self.logs += text
        # Only keep the last 3000 characters so the browser doesn't freeze
        self.st_element.code(self.logs[-3000:], language="bash")
        
    def flush(self):
        pass

# Initialize the chat log from the JSON file
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat()

# Render previous messages
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input Box
if prompt := st.chat_input("Type your requirements or answer the PM's questions..."):
    # Save and display user input
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    save_chat(st.session_state.chat_history)
    st.chat_message("user").write(prompt)

    # Call your local AMD GPU Qwen model to act as the PM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            system_instruction = {
                "role": "system", 
                "content": (
                    "You are a Lead Technical Product Manager specializing in high-performance game architecture. "
                    "Your goal is to extract a bulletproof technical specification for projects requiring custom state machines and low-latency data flows. "
                    "Ask probing, clarifying questions specifically about: "
                    "1. Game loop execution order and tick rates. "
                    "2. Network and input latency mitigation strategies. "
                    "3. Exact state machine transitions and edge cases. "
                    "4. Memory management and data structures. "
                    "Do not write code. Ask only 1 or 2 targeted questions at a time to refine the user's vision before finalizing the spec."
                )
            }
            
            payload = {
                "model": "qwen2.5-coder:14b",
                "messages": [system_instruction] + st.session_state.chat_history,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/chat", json=payload).json()
            pm_reply = response["message"]["content"]
            
            st.write(pm_reply)
            
            # Save the PM's response to the JSON file
            st.session_state.chat_history.append({"role": "assistant", "content": pm_reply})
            save_chat(st.session_state.chat_history)

# Sidebar controls
with st.sidebar:
    st.header("Pipeline Controls")
    st.write("Chat with the PM in the main window to refine your spec. Once approved, click below.")
    
    if st.button("🚀 Approve Spec & Deploy Dev Team", type="primary", use_container_width=True):
        
        st.write("### Live Agent Activity")
        log_window = st.empty() 
        
        # Hijack the terminal output
        old_stdout = sys.stdout
        sys.stdout = StreamlitConsole(log_window)
        
        try:
            # Compress the chat history into a single string
            full_spec_log = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
            
            # Trigger CrewAI in main.py
            final_result = run_pipeline(full_spec_log)
            
            st.success("Build Complete! Files saved and pushed to GitHub.")
            st.expander("View Final Agent Output").write(final_result)
        finally:
            sys.stdout = old_stdout
            
    if st.button("Clear Chat"):
        # Reset memory in the app and the file
        st.session_state.chat_history = [{"role": "assistant", "content": "I am your Lead PM. What are we building today?"}]
        save_chat(st.session_state.chat_history)
        st.rerun()

    st.divider()
    
    # Live File Explorer
    st.header("📁 Generated Code")
    
    # Scan the target directories for generated files
    target_dirs = ["docs", "src", "tests"]
    found_files = []
    
    for d in target_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for file in files:
                    # Ignore python cache files
                    if not file.endswith(".pyc"):
                        found_files.append(os.path.join(root, file))
    
    if found_files:
        selected_file = st.selectbox("Select a file to view:", ["-- Select --"] + found_files)
        
        if selected_file != "-- Select --":
            with open(selected_file, "r", encoding="utf-8") as f:
                file_content = f.read()
                
            # Determine syntax highlighting
            lang = "python" if selected_file.endswith(".py") else "markdown"
            
            st.markdown(f"**{selected_file}**")
            st.code(file_content, language=lang)
    else:
        st.info("No generated files found yet. Deploy the dev team to generate code!")