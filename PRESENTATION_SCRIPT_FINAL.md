# Sarika: Election Assistance Chatbot - Final Presentation Script

**Presenter Name:** [Your Name]
**Project Title:** Sarika - AI-Powered Election Assistance Chatbot
**Duration:** ~10-15 Minutes

---

## Chapter I: Introduction of the Project

**Slide 1: Title Slide**
"Namaste and Good Morning. My name is [Your Name], and today I am presenting my project: **Sarika**, an AI-powered election assistance chatbot designed for the voters of Nepal."

**Slide 2: Problem Background**
"Let's start with the problem we are solving.
The election process in Nepal is governed by complex legal acts and dispersed guidelines. For a common voter, finding specific answers—like 'Where is my polling center?' or 'What documents do I need?'—is incredibly difficult.
Information is buried in 50-page PDF files or scattered across different government websites. This 'Information Gap' leads to voter confusion and, in some cases, disenfranchisement."

**Slide 3: Identifying Your Users**
"Our target users are the 1.7 Crore eligible voters of Nepal.
*   **The First-Time Voter:** Who doesn't know the registration process.
*   **The Rural Voter:** Who needs simple, direct answers without navigating complex portals.
*   **The Elderly/Disabled:** Who need information on accessibility.
Their expectation is simple: They want an immediate, accurate answer without reading a legal manual."

**Slide 4: Project Objectives**
"The primary objectives of this project were:
1.  **To Democratize Information:** Make election laws accessible to everyone.
2.  **To Ensure Accuracy:** Unlike generic AI, our system must not 'hallucinate' or invent facts.
3.  **To Optimize Performance:** The system must run efficiently on standard hardware with sub-second response times."

---

## Chapter II: Literature Review

**Slide 5: Existing System Review**
"Before building Sarika, we analyzed existing solutions:
1.  **Traditional Keyword Search:** (Ctrl+F) This fails if you don't know the exact legal term. If you search 'voting place' but the document says 'polling center', you get zero results.
2.  **Generic LLMs (ChatGPT):** These are powerful but dangerous for legal tasks. They often 'hallucinate,' confidentially providing incorrect dates or procedures."

**Slide 6: Algorithm Selection & Mathematical Justification**
"To solve this, we selected a **Hybrid Search Algorithm**. We combine two mathematical approaches to get the best of both worlds."

**Slide 7: Mathematical Formulation (The 'Why')**
"First, we use **TF-IDF (Term Frequency-Inverse Document Frequency)** for keyword precision.
*   *Mathematically:* It calculates a score based on how rare a word is.
    $$ W_{x,y} = tf_{x,y} \times \log(\frac{N}{df_x}) $$
    *   If a user types a specific district name like 'Kathmandu', TF-IDF gives it a high weight because it's a rare, specific token.*

Second, we use **Cosine Similarity with Vector Embeddings** for semantic meaning.
*   *Mathematically:* We represent text as 384-dimensional vectors. We calculate the angle between the User Query Vector ($A$) and the Document Vector ($B$):
    $$ \text{Similarity} = \cos(\theta) = \frac{A \cdot B}{||A|| ||B||} $$
    *   This allows the system to understand that 'voting place' and 'polling center' are mathematically close (parallel vectors), even if they share no words.*

Our final algorithm combines these: $\text{Final Score} = \alpha \cdot \text{VectorScore} + (1-\alpha) \cdot \text{KeywordScore}$."

---

## Chapter III: Project Methodology

**Slide 8: Development Methodology**
"We followed an **Incremental Development Methodology**.
*   **Why?** AI development is experimental. We started with a simple keyword search. Once that worked, we added Vector Search. Finally, we added the Hybrid logic and caching layers. This allowed us to test and refine each component independently."

**Slide 9: System Architecture**
"Here is the high-level workflow of Sarika:
1.  **User Interface:** The user asks a question via the web widget.
2.  **API Gateway (FastAPI):** The request hits our Python backend.
3.  **Processing Layer:**
    *   The query is sent to **LangChain**.
    *   It runs in parallel through **FAISS** (Vector Store) and **Scikit-Learn** (TF-IDF).
4.  **Retrieval:** The top 3 most relevant chunks are retrieved.
5.  **Response:** The system returns the exact text from the official Knowledge Base."

---

## Chapter IV: About AI & Machine Learning Implementation

**Slide 10: Dataset Description**
"Our dataset is a custom-built **Knowledge Base**.
*   **Source:** Official Election Commission of Nepal documents (Acts, Rules, FAQs).
*   **Format:** Unstructured text files (`.txt`).
*   **Volume:** We processed key legal documents, broken down into manageable 'chunks' of text."

**Slide 11: Pre-processing & Feature Engineering**
"Data pre-processing was critical:
1.  **Text Splitting:** We used `RecursiveCharacterTextSplitter` to break long laws into 1000-character chunks with overlap, ensuring context isn't lost.
2.  **Normalization:** We convert everything to lowercase and remove noise.
3.  **Embedding:** We pass these chunks through the `all-MiniLM-L6-v2` model to generate dense vector representations."

**Slide 12: Model Training & Evaluation**
"Since we are using a pre-trained embedding model (BERT-based), we didn't need 'training' in the traditional sense.
*   **Indexing:** Instead, we performed 'Indexing'. We built a FAISS index which acts as our model's memory.
*   **Evaluation:** We used qualitative evaluation—testing the bot with 50+ real-world questions to verify if the retrieved answers matched the legal facts."

---

## Chapter V: Final Outcome of the Project

**Slide 13: The Final Product**
"The result is a fully functional, web-based chatbot.
*   **Speed:** It responds in under 800 milliseconds.
*   **Accuracy:** It successfully retrieves specific clauses (e.g., 'Form 102') and general concepts.
*   **Interface:** It features a modern 'Glassmorphism' UI that is responsive on mobile and desktop."

*[Show Screenshot or Demo Video Here]*

---

## Chapter VI: Conclusion and Future Enhancements

**Slide 14: Conclusion**
"In conclusion, Sarika successfully demonstrates that **Retrieval-Augmented Generation (RAG)** is a viable solution for government tech. We bridged the gap between complex legal data and the common citizen without relying on expensive or unreliable external AI services."

**Slide 15: Limitations & Future Work**
"**Limitations:**
*   Currently, it only supports English.
*   It is text-based only.

**Future Enhancements:**
1.  **NLP for Nepali:** Implementing Devanagari script support.
2.  **Voice Interface:** Allowing illiterate voters to ask questions verbally.
3.  **Admin Dashboard:** To visualize which questions are trending across different districts."

---

## Chapter VII: Key Takeaways from the Course

**Slide 16: Key Takeaways**
"Reflecting on this course, my key takeaways are:
1.  **The Power of Embeddings:** I learned how converting text to numbers (Vectors) unlocks the ability for computers to 'understand' meaning.
2.  **Hybrid Systems are Superior:** Relying on just one algorithm (like only Keywords or only Vectors) is rarely enough; combining them yields the best results.
3.  **Real-World AI:** I learned that 80% of AI work is data preparation and pipeline engineering, not just model training.
4.  **Deployment:** I gained practical skills in FastAPI and integrating Python AI backends with web frontends."

**Slide 17: Thank You**
"Thank you for your time. I am now open to any questions."
