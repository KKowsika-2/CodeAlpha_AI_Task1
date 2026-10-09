# Project Explanation & Defense Guide 🎓

**Project Title**: FAQ Chatbot  
**Repository Name**: CodeAlpha_FAQChatbot  
**Program**: CodeAlpha Artificial Intelligence Internship  

---

## 1. Problem Statement
Organizations, internship programs, and educational platforms receive hundreds of repetitive inquiry emails every day regarding basic technical setups, project guidelines, and domain concepts. Manually answering these questions is inefficient and slows down student progress.

---

## 2. Proposed Solution
We developed a local, lightweight, AI-driven FAQ Chatbot web application. The chatbot automatically matches incoming user questions against a structured dataset of Frequently Asked Questions (`faqs.json`) and returns the most relevant answer instantaneously, alongside a similarity match score indicator.

---

## 3. System Workflow Architecture

```text
  [ User Query ]
        │
        ▼
  [ Preprocessing ] ───► (Lowercasing, Regex Special Char Stripping, Stop Words Filter)
        │
        ▼
  [ TF-IDF Vectorizer ] ──► Transforms clean text into numerical TF-IDF Vector
        │
        ▼
  [ Cosine Similarity ] ──► Calculates Similarity Score against pre-vectorized FAQ dataset
        │
        ▼
  ┌───────────────────────────┐
  │ Score >= Threshold (0.25) │
  └─────────────┬─────────────┘
                │
         ┌──────┴──────┐
         │             │
        YES            NO
         │             │
         ▼             ▼
  [ Return Best ]  [ Return Polite ]
  [ FAQ Answer  ]  [ Fallback Msg  ]
```

---

## 4. Algorithm Explanation

### A. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF converts text into meaningful numerical vectors:
- **TF (Term Frequency)**: Measures the frequency of a word $t$ in document $d$.
- **IDF (Inverse Document Frequency)**: Assigns higher weight to rare, informative words while scaling down ubiquitous words like `"the"` or `"is"`.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

### B. Cosine Similarity
Calculates the geometric cosine of the angle between two vectors in a high-dimensional vector space:

$$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

- Returns `1.0` for identical orientation and `0.0` for orthogonal (unrelated) vectors.

---

## 5. Advantages & Limitations

### Key Advantages
1. **100% Offline & Free**: Requires no external APIs, subscriptions, or internet connectivity once installed.
2. **Deterministic & Fast**: Instant execution with zero latency or hallucination risks.
3. **Transparent**: Displays match confidence scores for full explainability.

### Limitations
1. Does not infer deep semantic relationships without exact or root word overlap.
2. Limited to predefined answers in `faqs.json`.

---

## 🎥 1-Minute Project Presentation & Video Script

> **[0:00 - 0:10] Introduction**  
> *"Hello everyone! My name is [Your Name], and today I am presenting Task 1 for my CodeAlpha Artificial Intelligence Internship: an AI FAQ Chatbot built using Python, Streamlit, and scikit-learn."*

> **[0:10 - 0:25] Live Demo - Matching FAQ**  
> *"As you can see on screen, our web app provides a clean chat interface. Let's test a question: 'What is Artificial Intelligence?'. When I press enter, the bot uses TF-IDF vectorization and Cosine Similarity to find the best match with a 100% confidence score."*

> **[0:25 - 0:40] Live Demo - Paraphrasing & Fallback**  
> *"Even if I paraphrase the question to 'Explain machine learning overfitting', the model correctly matches it with a high similarity score. If I ask an out-of-domain question like 'How to bake a cake?', the bot detects that the similarity score is below our 25% threshold and politely informs me it doesn't know."*

> **[0:40 - 1:00] Conclusion & Tech Summary**  
> *"The app also features sample question pills, threshold customization, and a clear chat button. It runs entirely offline on local hardware without paid APIs. Thank you CodeAlpha for this opportunity!"*
