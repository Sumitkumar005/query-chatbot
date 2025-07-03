import streamlit as st
import sqlite3
import pandas as pd
import os
import datetime
import pytz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from tavily import TavilyClient
import getpass
from dotenv import load_dotenv
import PyPDF2
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Get API keys
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" in os.environ:
        GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    else:
        st.error("Google Gemini API key not found. Add it to '.streamlit/secrets.toml' or set it as an env variable.")
        st.stop()
except Exception as e:
    st.error(f"Error accessing Google API key: {str(e)}. Check your setup.")
    st.stop()

try:
    if "TAVILY_API_KEY" in st.secrets:
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    elif "TAVILY_API_KEY" in os.environ:
        TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
    else:
        TAVILY_API_KEY = getpass.getpass("Enter TAVILY_API_KEY: ")
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
except Exception as e:
    st.error(f"Error accessing Tavily API key: {str(e)}. Check your setup.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Supported languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}

# Initialize SQLite database only if it doesn't exist
def init_db_if_needed():
    if not os.path.exists("data/chatbot.db"):
        conn = sqlite3.connect("data/chatbot.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                university TEXT,
                program TEXT,
                tuition INTEGER,
                location TEXT,
                visa_service TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

# Scrape website and save to /scraped_data
def scrape_website(url, keep_old_data):
    try:
        if not url.startswith("http"):
            st.error("Invalid URL. Please include 'http://' or 'https://'.")
            return False
        
        os.makedirs("scraped_data", exist_ok=True)
        if not keep_old_data:
            for file in os.listdir("scraped_data"):
                os.remove(os.path.join("scraped_data", file))
            st.write("Cleared old scraped data.")
        
        with st.spinner(f"Scraping {url}..."):
            crawl_results = tavily_client.crawl(url=url, max_depth=3, extract_depth="advanced")
        
        if not crawl_results["results"]:
            st.error("No data scraped. Check URL or Tavily API limits.")
            return False
        
        for i, result in enumerate(crawl_results["results"]):
            content = result.get("raw_content", "")
            if not content.strip():
                st.warning(f"Page {i+1} has no content. Skipping.")
                continue
            file_path = f"scraped_data/page_{i+1}_{url.replace('https://', '').replace('/', '_')}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            st.write(f"Saved {len(content)} characters to {file_path}")
        
        st.success(f"Scraped {len(crawl_results['results'])} pages from {url}")
        return True
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        return False

# Load file into appropriate storage
def load_file(file):
    try:
        file_name = file.name
        file_ext = file_name.split('.')[-1].lower()
        
        os.makedirs("data", exist_ok=True)  # Ensure data directory exists
        
        if file_ext in ['csv', 'xlsx']:
            init_db_if_needed()  # Create database if it doesn't exist
            conn = sqlite3.connect("data/chatbot.db")
            if file_ext == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            expected_columns = ['university', 'program', 'tuition', 'location', 'visa_service']
            if not all(col in df.columns for col in expected_columns):
                st.error(f"File {file_name} must have columns: {', '.join(expected_columns)}")
                return False
            
            df.to_sql('universities', conn, if_exists='append', index=False)
            conn.close()
            file_path = f"data/{file_name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Loaded {len(df)} rows from {file_name} into universities table and saved to {file_path}")
        
        elif file_ext == 'sql':
            init_db_if_needed()  # Create database if it doesn't exist
            conn = sqlite3.connect("data/chatbot.db")
            sql_content = file.read().decode('utf-8')
            try:
                conn.executescript(sql_content)
                conn.commit()
                st.success(f"Executed SQL from {file_name}")
            except Exception as e:
                st.error(f"Error executing SQL from {file_name}: {str(e)}")
                return False
            conn.close()
            file_path = f"data/{file_name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Saved SQL file to {file_path}")
        
        elif file_ext == 'pdf':
            os.makedirs("scraped_data", exist_ok=True)
            timestamp = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y%m%d_%H%M')
            text_file = f"scraped_data/pdf_upload_{timestamp}.txt"
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            if not text.strip():
                st.error(f"No text extracted from {file_name}")
                return False
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            st.success(f"Saved PDF text to {text_file}")
        
        elif file_ext == 'txt':
            file_path = "data/university_info.txt" if file_name == "university_info.txt" else \
                       f"scraped_data/txt_upload_{datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y%m%d_%H%M')}.txt"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Saved text file to {file_path}")
        
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return False
        
        return True
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return False

# Delete file or database
def delete_file(file_name):
    try:
        if file_name == "chatbot.db":
            file_path = "data/chatbot.db"
            if os.path.exists(file_path):
                os.remove(file_path)
                st.write(f"Deleted {file_path}")  # Log deletion
                # Prevent immediate recreation by not calling init_db() here
                return True
        elif file_name == "university_info.txt":
            file_path = "data/university_info.txt"
        elif file_name.endswith('.txt'):
            file_path = f"scraped_data/{file_name}"
        else:
            file_path = f"data/{file_name}"
        
        if os.path.exists(file_path):
            os.remove(file_path)
            st.write(f"Deleted {file_path}")  # Log deletion
            return True
        else:
            st.error(f"File {file_path} not found.")
            return False
    except Exception as e:
        st.error(f"Error deleting {file_name}: {str(e)}")
        return False

# Load and index data for FAISS (optional, only for text data)
def load_and_index_data():
    all_texts = []
    
    # Load university_info.txt
    uni_info_file = "data/university_info.txt"
    if os.path.exists(uni_info_file):
        with open(uni_info_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                all_texts.append(content)
                st.write(f"Loaded {len(content)} characters from university_info.txt")
    
    # Load scraped_data
    scraped_dir = "scraped_data"
    if os.path.exists(scraped_dir):
        for file_name in os.listdir(scraped_dir):
            if file_name.endswith(".txt"):
                with open(os.path.join(scraped_dir, file_name), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        all_texts.append(content)
                        st.write(f"Loaded {len(content)} characters from {file_name}")
    
    if not all_texts:
        st.write("No text data found for indexing. SQL will be used for structured queries.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in all_texts:
        chunks.extend(text_splitter.split_text(text))
    st.write(f"Created {len(chunks)} chunks for indexing from {len(all_texts)} sources.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            st.write("Cleared old FAISS index.")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local("vectorstore")
        st.write("Created new FAISS index.")
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Failed to create FAISS index: {str(e)}")
        return None

# Generate SQL query using Gemini
def generate_sql_query(query, lang_code):
    if lang_code != "en":
        try:
            query = GoogleTranslator(source=lang_code, target="en").translate(query)
            st.write(f"Translated query from {lang_code} to en: {query}")
        except Exception as e:
            st.error(f"Translation failed: {str(e)}. Using original query.")
    
    prompt = f"""
You are an expert SQL assistant for a university and visa chatbot. Convert the user's natural language query into a valid SQLite SELECT query based on the 'universities' table schema. The table has the following columns:

- university (TEXT)
- program (TEXT)
- tuition (INTEGER)
- location (TEXT)
- visa_service (TEXT)

Rules:
1. Generate only the SQL SELECT query (no explanations or additional text).
2. Use the exact table name 'universities'.
3. If the query cannot be answered with the available data, return: 'No relevant data in database.'
4. Handle ambiguous queries by assuming they refer to the universities table.
5. Ensure the query is syntactically correct for SQLite.

Examples:
- User: "What is the tuition at Sample University?"
  SQL: SELECT tuition FROM universities WHERE university = 'Sample University';
- User: "Which universities offer Computer Science?"
  SQL: SELECT university FROM universities WHERE program = 'Computer Science';
- User: "Tell me about visamonk.ai"
  SQL: No relevant data in database.

User query: {query}
SQL query:
"""
    try:
        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        if not sql_query.startswith('SELECT') and 'No relevant data' not in sql_query:
            return 'Invalid query', []
        return sql_query, generate_follow_ups(query, st.session_state.chat_history)
    except Exception as e:
        return f"Error generating SQL: {str(e)}", []

# Execute SQL query
def execute_sql_query(sql_query):
    try:
        conn = sqlite3.connect("data/chatbot.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        if not results:
            return "No data found.", []
        
        if len(results) == 1 and len(columns) == 1:
            response = str(results[0][0])
        else:
            response = "\n".join([f"{', '.join([f'{col}: {val}' for col, val in zip(columns, row)])}" for row in results])
        return response, []
    except Exception as e:
        return f"Error executing SQL: {str(e)}", []

# Retrieve relevant history
def retrieve_relevant_history(query, limit=3):
    if not os.path.exists("data/chatbot.db"):
        return []
    conn = sqlite3.connect("data/chatbot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT query, response FROM conversation_history ORDER BY id DESC LIMIT 50")
    history = cursor.fetchall()
    conn.close()
    
    if not history:
        return []
    
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    query_embedding = embeddings_model.embed_query(query)
    history_embeddings = [embeddings_model.embed_query(h[0]) for h in history]
    
    similarities = cosine_similarity([query_embedding], history_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:limit]
    
    relevant = [history[i] for i in top_indices]
    return relevant

# Generate follow-up questions
def generate_follow_ups(query, conversation_history):
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a, _, _ in conversation_history[-3:]])
    prompt = f"""
Based on the following conversation history and the latest user query, generate 3 relevant follow-up questions that the user might ask next to get more information. The questions should be directly related to the context of the conversation and the latest query.

Conversation history:
{history_text}

Latest query: {query}

Provide only the follow-up questions as a list, without any introductory text.
"""
    try:
        response = model.generate_content(prompt)
        questions = [line.strip() for line in response.text.split('\n') if line.strip()]
        return questions[:3]
    except Exception:
        return ["What other programs are available at this university?", "Can you tell me more about the visa services?", "What is the location of this university?"]

# RAG pipeline for non-SQL queries
def rag_query(query, retriever, lang_code):
    if retriever is None:
        return "No text data available. Please use structured queries with SQL.", []
    
    translated_query = query
    if lang_code != "en":
        try:
            translated_query = GoogleTranslator(source=lang_code, target="en").translate(query)
            st.write(f"Translated query from {lang_code} to en: {translated_query}")
        except Exception as e:
            st.error(f"Translation failed: {str(e)}. Using original query.")
            translated_query = query
    
    relevant_history = retrieve_relevant_history(translated_query)
    history_context = "\n".join([f"Past Q: {h[0]}\nPast A: {h[1]}" for h in relevant_history])
    
    docs = retriever.get_relevant_documents(translated_query)
    context = "\n".join([doc.page_content for doc in docs])
    
    full_context = f"{history_context}\n\n{context}"
    
    is_concise = any(phrase in query.lower() for phrase in ["in one line", "briefly", "shortly"])
    concise_instruction = "Provide a very concise answer in one line." if is_concise else "Provide a concise answer."
    
    prompt = f"""
You are a helpful assistant for university and visa information. {concise_instruction} Use the context provided, which includes past interactions and current knowledge. If the answer isn’t in the context, say 'I don’t have enough information.'

Context: {full_context}

Question: {translated_query}
Answer:
"""
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        if lang_code != "en":
            try:
                response_text = GoogleTranslator(source="en", target=lang_code).translate(response_text)
                st.write(f"Translated response from en to {lang_code}: {response_text}")
            except Exception as e:
                st.error(f"Translation to {lang_code} failed: {str(e)}. Returning English response.")
        follow_ups = generate_follow_ups(translated_query, st.session_state.chat_history)
        if lang_code != "en":
            try:
                follow_ups = [GoogleTranslator(source="en", target=lang_code).translate(q) for q in follow_ups]
            except Exception as e:
                st.error(f"Follow-up translation failed: {str(e)}. Returning English follow-ups.")
        return response_text, follow_ups
    except Exception as e:
        return f"Error generating response: {str(e)}", []

# Text-to-Speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
    .timestamp {
        color: gray;
        font-size: 0.8em;
        margin-top: 8px;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Create data and scraped_data folders
os.makedirs("data", exist_ok=True)
os.makedirs("scraped_data", exist_ok=True)
init_db_if_needed()

# Sidebar
with st.sidebar:
    st.write("👋 Welcome to University Chatbot!")
    
    st.session_state.language = st.selectbox("Select Language", list(LANGUAGES.keys()))
    
    st.subheader("Admin Access")
    admin_password = st.text_input("Enter admin password:", type="password")
    if st.button("Login"):
        if admin_password == "admin123":
            st.session_state.admin_authenticated = True
            st.success("Admin access granted!")
        else:
            st.session_state.admin_authenticated = False
            st.error("Incorrect password.")
    
    if st.session_state.admin_authenticated:
        st.subheader("Admin: Scrape Website")
        scrape_url = st.text_input("Enter URL to scrape (e.g., https://www.visamonk.ai/):", value="https://www.visamonk.ai/")
        data_option = st.radio("Data Handling:", ("Keep old scraped data", "Replace with new data"))
        keep_old_data = data_option == "Keep old scraped data"
        if st.button("Scrape Now"):
            with st.spinner("Scraping website..."):
                if scrape_website(scrape_url, keep_old_data):
                    st.rerun()
        
        st.subheader("Admin: Upload Files")
        uploaded_file = st.file_uploader("Upload CSV, XLSX, SQL, PDF, or TXT", type=['csv', 'xlsx', 'sql', 'pdf', 'txt'])
        if uploaded_file and st.button("Upload File"):
            if load_file(uploaded_file):
                st.rerun()
        
        st.subheader("Delete Files/Database")
        all_files = []
        if os.path.exists("data/university_info.txt"):
            all_files.append("university_info.txt")
        if os.path.exists("data/chatbot.db"):
            all_files.append("chatbot.db")
        all_files += [f for f in os.listdir("scraped_data") if f.endswith('.txt')]
        all_files += [f for f in os.listdir("data") if f.endswith(('.csv', 'xlsx', 'sql'))]
        if all_files:
            file_to_delete = st.selectbox("Select file or database to delete:", all_files)
            if st.button("Delete File/Database"):
                if delete_file(file_to_delete):
                    st.rerun()
        else:
            st.write("No files or database to delete.")
        
        if st.button("Re-Index Data"):
            with st.spinner("Re-indexing data..."):
                retriever = load_and_index_data()
                if retriever:
                    st.success("Data re-indexed successfully!")
                else:
                    st.write("No text data to index. SQL will handle structured queries.")
        
        if st.button("List Current Files"):
            st.write("**Files in data/:**")
            data_files = [f for f in os.listdir("data") if f != "chatbot.db"]
            for f in data_files:
                st.write(f"- {f}")
            st.write("**Files in scraped_data/:**")
            scraped_files = [f for f in os.listdir("scraped_data") if f.endswith('.txt')]
            for f in scraped_files:
                st.write(f"- {f}")
    else:
        st.write("Admins only: Enter password to access admin features.")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.feedback = {}
        st.rerun()
    
    st.subheader("Example Questions")
    examples = [
        "What is the tuition at Sample University?",
        "Which universities offer Computer Science?",
        "What is visamonk.ai?"
    ]
    for q in examples:
        if st.button(q):
            st.session_state.current_question = q
            st.rerun()

# Main interface
st.title("🎓 University Chatbot")
chat_container = st.container()

for i, (user_msg, bot_msg, timestamp, follow_ups) in enumerate(st.session_state.chat_history):
    with chat_container:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
            st.caption(timestamp)
        with col2:
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {bot_msg}</div>', unsafe_allow_html=True)
            st.caption(timestamp)
            if st.button("Play Response", key=f"play_{i}"):
                audio_fp = text_to_speech(bot_msg)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button("👍", key=f"up_{i}"):
                    st.session_state.feedback[i] = "positive"
                    st.success("Thank you for your feedback!")
            with feedback_col2:
                if st.button("👎", key=f"down_{i}"):
                    st.session_state.feedback[i] = "negative"
                    st.success("Thank you for your feedback!")
            if i in st.session_state.feedback:
                st.write(f"You {'liked' if st.session_state.feedback[i] == 'positive' else 'disliked'} this response")
            if follow_ups:
                st.write("**Follow-up Questions:**")
                for j, follow_up in enumerate(follow_ups):
                    if st.button(follow_up, key=f"follow_up_{i}_{j}"):
                        st.session_state.current_question = follow_up
                        st.rerun()

input_col1, input_col2 = st.columns([4, 1])
with input_col1:
    user_input = st.text_input("Ask your question:", value=st.session_state.current_question, key="input", placeholder="e.g., What is the tuition at Sample University?")
with input_col2:
    send_button = st.button("Send 📤")

if send_button and user_input:
    retriever = load_and_index_data()
    sql_query, sql_follow_ups = generate_sql_query(user_input, LANGUAGES[st.session_state.language])
    if sql_query.startswith('SELECT'):
        response, _ = execute_sql_query(sql_query)
        if response != "No data found." and response != "Error executing SQL: ":
            follow_ups = sql_follow_ups or generate_follow_ups(user_input, st.session_state.chat_history)
            st.write("Used SQL query.")
        else:
            if retriever:
                response, follow_ups = rag_query(user_input, retriever, LANGUAGES[st.session_state.language])
                st.write("Used RAG query (fallback from SQL).")
            else:
                response = "No structured or text data available for this query."
                follow_ups = []
                st.write("Used SQL, but no data found. No text index available.")
    else:
        if retriever:
            response, follow_ups = rag_query(user_input, retriever, LANGUAGES[st.session_state.language])
            st.write("Used RAG query.")
        else:
            response = "No text data available. Please use structured queries with SQL."
            follow_ups = []
            st.write("No text index available. SQL attempted but failed.")

    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M')
    st.session_state.chat_history.append((user_input, response, current_time, follow_ups))
    
    if os.path.exists("data/chatbot.db"):
        conn = sqlite3.connect("data/chatbot.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversation_history (query, response, timestamp) VALUES (?, ?, ?)", 
                       (user_input, response, current_time))
        conn.commit()
        conn.close()
    
    st.session_state.current_question = ""
    st.rerun()