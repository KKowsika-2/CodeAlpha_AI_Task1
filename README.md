# CodeAlpha_FAQChatbot 🤖

An intelligent, lightweight, local FAQ Chatbot web application built for the **CodeAlpha Artificial Intelligence Internship Task 1**.

This application accepts questions from users and retrieves the most relevant answer from a curated database of Frequently Asked Questions (FAQs) using **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, and **Cosine Similarity**.

---

## 🌟 Key Features

- **Intuitive Streamlit UI**: Modern chat message bubbles, clean sidebar controls, and responsive layout.
- **Local & Offline Execution**: Runs 100% locally on standard computer hardware with zero API keys or paid services.
- **TF-IDF & Cosine Similarity Engine**: Natural language text matching using `scikit-learn`.
- **Match Confidence Indicator**: Visual badges displaying the exact matching similarity score and matched FAQ question.
- **Customizable Similarity Threshold**: Adjustable threshold slider (default 0.25) to control answer precision.
- **25+ Curated FAQs**: Covers AI/ML fundamentals, Python programming, Git/GitHub, Streamlit, CodeAlpha tasks, and troubleshooting.
- **Interactive Quick-Sample Buttons**: One-click preset questions to test chatbot capabilities instantly.
- **Session State History**: Persists chat messages throughout the session with a **Clear Chat** button to reset history.
- **Graceful Error Handling**: Handles blank inputs, out-of-domain queries, and dataset errors safely.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web application framework & chat UI |
| **scikit-learn** | `TfidfVectorizer` and `cosine_similarity` computation |
| **NumPy** | Vector score processing |
| **JSON** | Portable FAQ database storage (`faqs.json`) |

---

## 📐 How TF-IDF and Cosine Similarity Work

### 1. Text Preprocessing
Before vectorization, both the user's question and the pre-stored FAQs undergo text cleaning:
- Lowercasing all words.
- Stripping special characters and non-alphanumeric noise.
- Removing common English stop words (e.g., `"what"`, `"is"`, `"the"`).

### 2. Term Frequency-Inverse Document Frequency (TF-IDF)
`TfidfVectorizer` transforms text into numerical vectors:
- **Term Frequency (TF)**: Measures how frequently a word appears in a specific document.
- **Inverse Document Frequency (IDF)**: Downweights common words across all documents and boosts rare, distinctive terms.

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 3. Cosine Similarity Matching
Cosine similarity computes the cosine of the angle between the user query vector $\vec{q}$ and each FAQ question vector $\vec{d}$:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

- Score range: `0.0` (completely dissimilar) to `1.0` (identical).
- If $\text{score} \ge \text{threshold}$, the chatbot returns the associated answer.
- If $\text{score} < \text{threshold}$, the chatbot responds with a polite fallback message.

---

## 📂 Project Structure

```text
CodeAlpha_FAQChatbot/
│
├── app.py                  # Main Streamlit application & similarity engine
├── faqs.json               # JSON dataset containing 26 curated FAQs
├── requirements.txt        # Python package dependencies
├── README.md               # Comprehensive project documentation
├── PROJECT_EXPLANATION.md  # Viva/Presentation guide & 1-min video script
└── .gitignore              # Specifies files to exclude from Git tracking
```

---

## 🚀 Installation & Setup Instructions

### Prerequisites
- Python 3.8 or higher installed on your system.
- Git installed (optional, for cloning).

### Step 1: Clone or Open the Repository
```bash
git clone https://github.com/YourUsername/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
```

### Step 2: Create a Virtual Environment
**Windows (PowerShell / Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run the Application

Execute the following command in your terminal (with the virtual environment activated):

```bash
streamlit run app.py
```

The Streamlit web interface will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Sample Questions & Expected Behavior

| Query Type | Input Example | Expected Outcome |
|---|---|---|
| **Exact Match** | `What is Artificial Intelligence (AI)?` | High Match score (~100%), returns exact AI definition. |
| **Paraphrased** | `Explain machine learning overfitting to me` | Moderate Match score (>70%), returns overfitting explanation. |
| **Short Keyword** | `TF-IDF` | Moderate Match score (>50%), returns TF-IDF explanation. |
| **Out of Domain** | `What is the recipe for baking chocolate cake?` | Below threshold (<25%), returns polite fallback message. |
| **Empty Input** | `"   "` | Warning prompt requesting a valid question. |

---

## ⚠️ Limitations

1. **Keyword Overlap Dependency**: Traditional TF-IDF relies on exact or overlapping root vocabulary and does not capture deep semantic synonyms (unlike Transformer embeddings).
2. **Static Knowledge Base**: Answers are limited to the questions defined in `faqs.json`.
3. **No Multi-turn Context Memory**: Each question is evaluated independently of previous turns.

---

## 🔮 Future Enhancements

- Integration of **Sentence Transformers (SBERT)** or dense vector embeddings for semantic matching.
- SQLite or MongoDB backend integration for dynamic FAQ management.
- User feedback buttons (Thumbs Up / Down) for automated accuracy logging.
- Multi-language support using translation models.

---

## 👨‍💻 Author & Acknowledgments

Developed for **CodeAlpha AI Internship Program**.
- **Task Title**: FAQ Chatbot
- **Track**: Artificial Intelligence
