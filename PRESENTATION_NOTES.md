# Sarika - Election Assistance Chatbot
## Presentation Documentation & Tech Stack

### 1. Project Overview
**Sarika** is a specialized AI-powered chatbot designed to provide accurate, verified information regarding the election process in Nepal. Unlike generic AI models that may "hallucinate" incorrect facts, Sarika operates on a strict **Knowledge Base (KB)** architecture, ensuring every answer is derived from official acts, FAQs, and election guidelines.

### 2. Problem Statement
*   **Information Gap:** Voters often struggle to find specific election details (polling centers, registration hours, ID requirements) buried in long PDF documents.
*   **Misinformation Risk:** General-purpose LLMs (like ChatGPT) can generate plausible but factually incorrect legal/procedural advice.
*   **Accessibility:** Official government portals can be complex and hard to navigate on mobile devices.

### 3. The Solution: "Sarika"
A high-performance, hybrid-search chatbot that combines:
*   **Semantic Understanding:** Understands the *meaning* of user questions, not just keywords.
*   **Precision Retrieval:** Uses verified government documents as the sole source of truth.
*   **Instant Access:** Floating Action Button (FAB) interface that lives on top of any web page.

---

### 4. Technology Stack

#### **Backend (The Core)**
*   **Language:** Python 3.11 (Selected for rich AI/ML ecosystem).
*   **Framework:** **FastAPI** (Chosen for high performance, async support, and auto-generated API docs).
*   **Server:** **Uvicorn** (ASGI server for lightning-fast request handling).

#### **AI & NLP Engine**
*   **Orchestration:** **LangChain** (Manages the flow between user input, document retrieval, and response generation).
*   **Embeddings Model:** **HuggingFace `all-MiniLM-L6-v2`** (Converts text into 384-dimensional vectors for semantic comparison).
*   **Vector Store:** **FAISS (Facebook AI Similarity Search)** (Enables millisecond-level similarity search across thousands of document chunks).
*   **Keyword Search:** **Scikit-learn TF-IDF** (Term Frequency-Inverse Document Frequency) to catch exact keyword matches (e.g., specific district names).
*   **Algorithm:** **Hybrid Search** (Combines FAISS results + TF-IDF results to ensure both conceptual and literal relevance).

#### **Frontend (The Interface)**
*   **Tech:** Vanilla **HTML5, CSS3, JavaScript (ES6+)**.
*   **Design:** Custom responsive UI with a "Glassmorphism" aesthetic.
*   **Reasoning:** No heavy frameworks (React/Vue) required for a lightweight widget; ensures maximum compatibility and zero build-step overhead for the UI.

#### **DevOps & Tools**
*   **Scripting:** PowerShell (`dev.ps1`) for automated environment setup and startup.
*   **Version Control:** Git & GitHub.
*   **Virtualization:** Python `venv` for isolated dependency management.

---

### 5. Key Technical Features

#### **A. Hybrid Search Architecture**
Most chatbots use *either* vector search (good for meaning) *or* keyword search (good for specific terms). Sarika uses **both**:
1.  **Vector Search (FAISS):** Finds concepts. (e.g., "Where do I vote?" matches "Polling Location").
2.  **Keyword Search (TF-IDF):** Finds entities. (e.g., "Kathmandu" or "Form 102").
3.  **Reranking:** The system combines scores from both engines to present the single best answer.

#### **B. Performance Optimization (Caching)**
*   **Hybrid Index Cache:** To avoid re-calculating heavy AI vectors on every restart, the system serializes the computed index to `data/hybrid_cache.pkl`.
*   **Result:** Startup time reduced from ~15 seconds to <1 second on subsequent runs.

#### **C. Zero-Hallucination Policy**
*   The system is explicitly coded **NOT** to generate text from scratch.
*   It retrieves *existing* text chunks from the `data/knowledge_base/` directory.
*   If no relevant information is found in the KB, it admits ignorance rather than inventing a fact.

---

### 6. Code Structure Overview

| File/Directory | Description |
| :--- | :--- |
| `app/main.py` | **The API Gateway.** Handles HTTP requests (`/answer`, `/feedback`) and serves the static frontend. |
| `app/chatbot.py` | **The Brain.** Contains the `ElectionChatbot` class, manages the FAISS index, executes the Hybrid Search logic. |
| `web/index.html` | **The Face.** Contains the chat widget UI, WebSocket-like polling logic, and the Floating Action Button. |
| `data/knowledge_base/` | **The Truth.** Folder containing `.txt` files (Acts, FAQs) that the bot "reads" to learn. |
| `dev.ps1` | **The Automator.** One-click script to install dependencies and start the server. |

---

### 7. Future Roadmap
*   **Multi-language Support:** Enabling Nepali language processing (Devanagari script support).
*   **Voice Interface:** Adding Speech-to-Text for accessibility.
*   **Admin Dashboard:** To view user queries and feedback analytics in real-time.
