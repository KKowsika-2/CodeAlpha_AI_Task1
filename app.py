import json
import os
import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="FAQ Chatbot | CodeAlpha",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.82rem;
        border-radius: 12px;
        font-weight: 600;
        margin-top: 6px;
    }
    .high-confidence {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #A5D6A7;
    }
    .medium-confidence {
        background-color: #FFFDE7;
        color: #F57F17;
        border: 1px solid #FFF59D;
    }
    .low-confidence {
        background-color: #FFEBEE;
        color: #C62828;
        border: 1px solid #EF9A9A;
    }
    .sample-btn {
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)


def preprocess_text(text: str) -> str:
    """Preprocess input text: lowercasing, stripping special characters and excess whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Strip leading/trailing/multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_data
def load_faqs(filepath: str = "faqs.json"):
    """Load FAQ dataset from JSON file."""
    if not os.path.exists(filepath):
        st.error(f"Error: FAQ data file '{filepath}' was not found.")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                st.error("Error: FAQ data format must be a list of FAQ objects.")
                return []
            return data
    except Exception as e:
        st.error(f"Error loading FAQ data: {str(e)}")
        return []


@st.cache_resource
def initialize_tfidf_model(faqs):
    """Build and fit TF-IDF Vectorizer on preprocessed FAQ questions."""
    if not faqs:
        return None, None
    
    questions = [preprocess_text(item["question"]) for item in faqs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix


def find_best_match(user_query: str, vectorizer, tfidf_matrix, faqs, threshold: float = 0.25):
    """Find the best matching FAQ answer using TF-IDF and cosine similarity."""
    cleaned_query = preprocess_text(user_query)
    
    if not cleaned_query:
        return {
            "answer": "Please enter a valid question.",
            "matched_question": None,
            "score": 0.0,
            "status": "empty"
        }
    
    # Vectorize user query
    query_vector = vectorizer.transform([cleaned_query])
    
    # Calculate cosine similarity with all stored FAQ vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    best_index = int(np.argmax(similarity_scores))
    best_score = float(similarity_scores[best_index])
    
    if best_score >= threshold:
        matched_faq = faqs[best_index]
        return {
            "answer": matched_faq["answer"],
            "matched_question": matched_faq["question"],
            "category": matched_faq.get("category", "General"),
            "score": round(best_score, 4),
            "status": "matched"
        }
    else:
        return {
            "answer": (
                "I'm sorry, I couldn't find a close answer to your question in my database. "
                "Try rephrasing your question or picking one of the suggested FAQs from the sidebar."
            ),
            "matched_question": faqs[best_index]["question"] if faqs else None,
            "score": round(best_score, 4),
            "status": "unmatched"
        }


# Initialize App State
faqs_data = load_faqs()
vectorizer, tfidf_matrix = initialize_tfidf_model(faqs_data)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI Internship FAQ Assistant. Ask me anything about Artificial Intelligence, Python, Machine Learning, or CodeAlpha internship tasks!",
            "matched_question": None,
            "score": None
        }
    ]

# Sidebar Controls & Information
with st.sidebar:
    st.title("🤖 Chatbot Controls")
    st.markdown("**CodeAlpha AI Task 1**")
    st.markdown("Powered by **TF-IDF** & **Cosine Similarity**")
    st.divider()

    # Threshold slider
    similarity_threshold = st.slider(
        "Similarity Threshold Score",
        min_value=0.10,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Minimum similarity score required to trigger an answer match."
    )
    
    st.divider()
    
    # Suggested Questions Section
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "What is Artificial Intelligence?",
        "What is Supervised Learning?",
        "How does TF-IDF work?",
        "What is a virtual environment in Python?",
        "How do I submit my CodeAlpha task?",
        "What is a .gitignore file?"
    ]
    
    selected_sample = None
    for q in sample_questions:
        if st.button(f"👉 {q}", key=f"sample_{q}", use_container_width=True):
            selected_sample = q

    st.divider()

    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help you today?",
                "matched_question": None,
                "score": None
            }
        ]
        st.rerun()

    st.markdown("---")
    st.caption("Total FAQs in Dataset: **{}**".format(len(faqs_data)))

# Main Chat Layout
st.markdown("<div class='main-header'>🤖 CodeAlpha FAQ Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>An intelligent local FAQ Assistant built with Python, Streamlit & scikit-learn.</div>", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display match metadata for assistant responses
        if msg.get("matched_question") and msg.get("score") is not None:
            score = msg["score"]
            percentage = round(score * 100, 1)
            
            if score >= 0.50:
                badge_class = "high-confidence"
                confidence_label = "High Match"
            elif score >= similarity_threshold:
                badge_class = "medium-confidence"
                confidence_label = "Moderate Match"
            else:
                badge_class = "low-confidence"
                confidence_label = "Below Threshold"

            st.markdown(
                f"""
                <div style="margin-top: 8px;">
                    <span class="confidence-badge {badge_class}">
                        Matched: "{msg['matched_question']}" ({percentage}% score - {confidence_label})
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

# Handle Input from Chat Input OR Sample Question Button
user_input = st.chat_input("Type your question here (e.g. 'What is TF-IDF?')...")

# Override with sample selection if clicked
if selected_sample:
    prompt_to_process = selected_sample
else:
    prompt_to_process = user_input

if prompt_to_process:
    clean_prompt = prompt_to_process.strip()
    
    if not clean_prompt:
        st.warning("Please type a non-empty question.")
    elif vectorizer is None or tfidf_matrix is None:
        st.error("FAQ dataset model failed to load. Please check `faqs.json`.")
    else:
        # Append User Message to State
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        
        # Get AI Response
        match_result = find_best_match(
            clean_prompt,
            vectorizer,
            tfidf_matrix,
            faqs_data,
            threshold=similarity_threshold
        )
        
        # Append Assistant Response to State
        st.session_state.messages.append({
            "role": "assistant",
            "content": match_result["answer"],
            "matched_question": match_result["matched_question"],
            "score": match_result["score"]
        })
        
        # Trigger UI Rerun
        st.rerun()
