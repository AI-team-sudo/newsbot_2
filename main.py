import streamlit as st
import os
import re
import time
from datetime import datetime
from pinecone import Pinecone
from deep_translator import GoogleTranslator
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("newsbot2")

STOPWORDS = {"news", "give", "me", "about", "on", "the", "is", "of", "for", "and", "with", "to", "in", "a"}

@st.cache_data(ttl=3600)
def extract_keywords(text):
    words = text.split()
    keywords = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(keywords)

@st.cache_data(ttl=3600)
def translate_to_gujarati(text):
    try:
        if re.search(r'[a-zA-Z]', text):
            return GoogleTranslator(source='en', target='gu').translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
    return text

@st.cache_data(ttl=3600)
def translate_content(text):
    try:
        return GoogleTranslator(source='en', target='gu').translate(text)
    except Exception as e:
        return f"Translation error: {e}"

@st.cache_data(ttl=3600)
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def highlight_keywords(text, keywords):
    if not text or not keywords:
        return text
    words = keywords.split()
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: yellow; color: black;">\1</mark>', text)

def format_display_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%d %b %Y")
    except Exception:
        return date_str

def search_namespace(namespace, query_embedding):
    try:
        results = index.query(
            vector=query_embedding,
            top_k=5,
            namespace=namespace,
            include_metadata=True
        )
        for match in results["matches"]:
            match["metadata"]["source"] = namespace
        return results["matches"]
    except Exception as e:
        st.error(f"Error searching in {namespace}: {e}")
        return []

def search_news(query):
    cleaned_query = extract_keywords(query)
    translated_query = translate_to_gujarati(cleaned_query)
    query_embedding = get_embedding(cleaned_query)

    namespaces = ["divyabhasker", "gujratsamachar"]
    all_results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_results = {executor.submit(search_namespace, ns, query_embedding): ns for ns in namespaces}
        for future in concurrent.futures.as_completed(future_results):
            all_results.extend(future.result())

    try:
        all_results.sort(
            key=lambda x: (
                datetime.strptime(x["metadata"]["date"], "%Y-%m-%d"),
                x["score"]
            ),
            reverse=True
        )
    except Exception as e:
        st.error(f"Error sorting results: {e}")
        all_results.sort(key=lambda x: x["score"], reverse=True)

    top_results = all_results[:5]
    return top_results, cleaned_query, translated_query

# Page configuration
st.set_page_config(page_title="Gujarati News Bot", page_icon="📰", layout="centered")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f2f5;
            color: black;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4CAF50;
            padding: 12px;
            border-radius: 8px;
            color: black !important;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white !important;
            padding: 12px;
            border-radius: 5px;
            border: none;
            width: 100%;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .news-card {
            background-color: #d9e2ec;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
            color: black;
        }
        .source-tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .divyabhasker {
            background-color: #ffeb3b;
            color: #000;
        }
        .gujratsamachar {
            background-color: #2196f3;
            color: white;
        }
        .read-more-button {
            display: inline-block;
            padding: 5px 10px;
            background-color: #333333;
            color: white !important;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
            margin-right: 10px;
        }
        .read-more-button:hover {
            background-color: #000000;
        }
        .translate-button {
            display: inline-block;
            padding: 5px 10px;
            background-color: #2196F3;
            color: white !important;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
        }
        .translate-button:hover {
            background-color: #1976D2;
        }
        .button-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .translated-content {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid #4CAF50;
        }
        .search-stats {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 14px;
        }
        .date-badge {
            background-color: #e0e0e0;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.markdown("<h1 style='text-align: center;'>📰 Gujarati News Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your query in English or Gujarati and get the latest news instantly.</p>", unsafe_allow_html=True)

# Initialize session state for translation toggles
if 'translation_states' not in st.session_state:
    st.session_state.translation_states = {}

# Search input
user_query = st.text_input("🔎 Enter your query (English or Gujarati):")
if st.button("Search News"):
    if user_query:
        start_time = time.time()
        with st.spinner("Fetching news... Please wait."):
            results, cleaned_query, translated_query = search_news(user_query)
        search_time = time.time() - start_time

        st.markdown(f"""
        <div class="search-stats">
            <p>🔍 Search completed in {search_time:.2f} seconds</p>
            <p>🔑 Search Keywords: <code>{cleaned_query}</code></p>
            {f"🌐 Gujarati Translation: <code>{translated_query}</code> 🇮🇳" if translated_query and translated_query != cleaned_query else ""}
        </div>
        """, unsafe_allow_html=True)

        if results:
            for idx, news in enumerate(results):
                metadata = news["metadata"]
                highlighted_title = highlight_keywords(metadata["title"], translated_query)
                highlighted_content = highlight_keywords(metadata["text"], translated_query)
                display_date = format_display_date(metadata["date"])

                translation_key = f"translate_{idx}"
                if translation_key not in st.session_state.translation_states:
                    st.session_state.translation_states[translation_key] = False

                source_class = metadata["source"].lower()
                st.markdown(f"""
                <div class="news-card">
                    <h3>{highlighted_title}</h3>
                    <p>
                        <span class="date-badge">📅 {display_date}</span>
                        <span class="source-tag {source_class}">{metadata['source']}</span>
                    </p>
                    <p>{highlighted_content}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"<a href='{metadata['link']}' target='_blank' class='read-more-button'>🔗 Read More</a>", unsafe_allow_html=True)
                with col2:
                    if st.button("🌐 Translate", key=translation_key):
                        st.session_state.translation_states[translation_key] = not st.session_state.translation_states[translation_key]

                if st.session_state.translation_states[translation_key]:
                    with st.spinner("Translating..."):
                        translated_content = translate_content(metadata["content"])
                        st.markdown(f"""
                        <div class="translated-content">
                            <h4>ગુજરાતી અનુવાદ:</h4>
                            <p>{translated_content}</p>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ No news found matching your query.")
