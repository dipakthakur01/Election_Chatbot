# Sarika: Election Assistance Chatbot - Presentation Outline

## Slide 1: Title Slide
- **Title:** Sarika: AI-Powered Election Assistance Chatbot
- **Subtitle:** Enhancing Voter Accessibility through Hybrid Search AI
- **Presenter:** [Your Name]
- **Context:** Election Technology / Civic Tech

## Slide 2: Introduction
- **The Hook:** Elections are the backbone of democracy, but information is often scattered and complex.
- **The Goal:** To bridge the gap between complex election data and the average voter.
- **What is Sarika?** A specialized, high-performance chatbot designed to provide instant, verified answers to election-related queries.

## Slide 3: The Problem
- **Information Overload:** Voters face hundreds of pages of PDFs, acts, and guidelines.
- **Accessibility Issues:** Official portals can be difficult to navigate on mobile.
- **Misinformation:** Generic AI (like ChatGPT) often "hallucinates" or gives incorrect legal advice.
- **The Need:** A system that is **Fast**, **Accurate**, and **Source-Verified**.

## Slide 4: The Solution
- **Knowledge-Base Driven:** Sarika only answers from official documents (FAQs, Acts, Guidelines).
- **Zero Hallucination:** If the answer isn't in the official docs, it doesn't invent one.
- **User-Centric:** Simple chat interface that lives on top of the website (Floating Action Button).

## Slide 5: System Architecture (How it Works)
- **1. Ingestion:** Text files (Election Acts) are loaded from the `data/knowledge_base` folder.
- **2. Chunking:** Large documents are split into smaller, meaningful segments.
- **3. Embedding:** The AI converts text into mathematical vectors (using HuggingFace models).
- **4. Retrieval:** When a user asks a question, the system searches these vectors to find the best match.
- **5. Response:** The most relevant text chunk is returned directly to the user.

## Slide 6: Key Innovation - "Hybrid Search"
- *Explain why this is better than standard search:*
- **Vector Search (FAISS):** Understands *meaning* (e.g., "Where to vote" matches "Polling Station").
- **Keyword Search (TF-IDF):** Captures *exact terms* (e.g., "Form 102", "Kathmandu").
- **The Result:** We combine both to ensure we never miss an answer, whether the user is vague or specific.

## Slide 7: Technology Stack
- **Backend:** Python 3.11, FastAPI (High performance).
- **AI/ML:** LangChain, HuggingFace (Embeddings), FAISS (Vector Store), Scikit-learn.
- **Frontend:** HTML5, CSS3, JavaScript (Lightweight, no frameworks needed).
- **DevOps:** PowerShell Automation, Git/GitHub.

## Slide 8: Performance & Optimization
- **Caching System:** We built a custom caching mechanism (`hybrid_cache.pkl`).
    - *Impact:* Startup time reduced from 15s to <1s.
- **Local Processing:** Runs entirely on the CPU (no expensive GPUs required).
- **Privacy:** No user data is sent to external cloud LLMs (like OpenAI).

## Slide 9: Live Demo
- *[Switch to Chatbot Interface]*
- **Demonstrate:**
    1.  Opening the bot via the Floating Action Button.
    2.  Asking a conceptual question: "How do I register to vote?"
    3.  Asking a specific keyword question: "What is the age limit?"
    4.  Showing the instant response speed.

## Slide 10: Future Roadmap
- **Multi-Language Support:** Adding full Devanagari (Nepali) support.
- **Voice Interaction:** Speech-to-Text for increased accessibility.
- **Admin Analytics:** A dashboard to see what voters are asking most frequently.

## Slide 11: Conclusion
- **Summary:** Sarika is a robust, cost-effective, and accurate solution for voter education.
- **Impact:** Empowers voters with the right information at the right time.
- **Closing:** Thank you!
- **Q&A**
