# Presentation Script for Sarika Chatbot

**Total Estimated Time:** 5-7 Minutes
**Tone:** Professional, Technical yet Accessible

---

### **Slide 1: Title Slide**
**(Speech)**
"Namaste and good morning everyone. My name is [Your Name], and today I am excited to present **Sarika**, an AI-powered election assistance chatbot.
In an era where digital information is everywhere, finding the *right* information about our democratic process is still surprisingly difficult. Sarika is designed to solve that problem."

---

### **Slide 2: Introduction**
**(Speech)**
"Elections are the backbone of our democracy. Every vote counts, but for a vote to count, the voter needs to be informed.
We asked ourselves a simple question: Why is it easier to order a pizza online than it is to find out exactly where your polling center is?
Our goal with Sarika was to bridge this gap—to take complex legal and procedural election data and make it instantly accessible to the average citizen."

---

### **Slide 3: The Problem**
**(Speech)**
"So, what is the actual problem we are solving?
1.  **Information Overload:** Currently, voters have to sift through hundreds of pages of PDF Acts and Guidelines to find a simple answer.
2.  **Accessibility:** Official government portals are comprehensive but can be hard to navigate, especially on mobile phones.
3.  **The AI Risk:** You might ask, 'Why not just use ChatGPT?' The problem is that generic AI 'hallucinates.' It might confidently tell you the wrong voting age or the wrong registration deadline because it's trained on the entire internet, not just Nepal's specific laws.
We needed a solution that was fast, easy, and most importantly, **100% accurate**."

---

### **Slide 4: The Solution**
**(Speech)**
"Enter **Sarika**.
Sarika is a specialized chatbot that operates on a strict **'Knowledge Base' architecture**.
Unlike ChatGPT, Sarika doesn't invent answers. It acts like a digital librarian. It only answers questions based on the official documents—Acts, FAQs, and Guidelines—that we have explicitly verified and fed into its system.
If the answer isn't in the official documents, Sarika will honestly say, 'I don't know,' rather than making something up. This ensures zero misinformation."

---

### **Slide 5: System Architecture**
**(Speech)**
"Let's look under the hood at how it works. The process follows five steps:
1.  **Ingestion:** We load the raw text files—the Election Acts and FAQs—into the system.
2.  **Chunking:** We break these large documents into smaller, meaningful paragraphs.
3.  **Embedding:** We use a HuggingFace model to convert this text into 'vectors'—essentially turning language into math that the computer can understand.
4.  **Retrieval:** When you ask a question, the system searches these vectors to find the most relevant paragraph.
5.  **Response:** That exact paragraph is delivered to you instantly."

---

### **Slide 6: Key Innovation - "Hybrid Search"**
**(Speech)**
"This is the most technical part of our project, and the one I'm most proud of. We use something called **Hybrid Search**.
Most search engines do one of two things:
1.  **Keyword Search:** They look for exact word matches (like 'Form 102').
2.  **Vector Search:** They look for meaning (like understanding that 'voting place' means 'polling center').
Sarika does **both**. We combine the precision of keyword matching with the intelligence of semantic understanding. This ensures that whether you ask a vague concept question or search for a specific legal term, you get the right answer."

---

### **Slide 7: Technology Stack**
**(Speech)**
"To build this, we used a robust modern tech stack:
*   **Backend:** Python 3.11 using FastAPI for high performance.
*   **AI Engine:** LangChain for orchestration and FAISS for our vector store.
*   **Frontend:** We kept the frontend lightweight with vanilla HTML and JavaScript, so it loads instantly on any device without heavy frameworks.
*   **DevOps:** We use PowerShell scripts for one-click automation."

---

### **Slide 8: Performance & Optimization**
**(Speech)**
"We also spent a lot of time on optimization.
AI models can be slow to load. To fix this, we built a custom **caching system**.
The first time you run Sarika, it calculates the math for all the documents. On every subsequent run, it loads from a cache file (`hybrid_cache.pkl`).
This reduced our startup time from 15 seconds down to **less than 1 second**, making the application practical for real-world deployment."

---

### **Slide 9: Live Demo**
**(Speech)**
"Enough talk, let me show you Sarika in action."
*[Switch to browser]*
"Here is the interface. You see the floating action button here.
Let's ask a conceptual question: *'How do I register to vote?'*
...and there is the answer, pulled directly from the Voter Registration Act.
Now let's try a specific detail: *'What is the age limit?'*
...and you see it instantly retrieves the specific clause regarding age eligibility.
The system is fast, responsive, and citation-backed."

---

### **Slide 10: Future Roadmap**
**(Speech)**
"We have built a solid foundation, but we aren't stopping here.
Our roadmap includes:
1.  **Multi-Language Support:** We want to add full Devanagari support so users can chat in Nepali.
2.  **Voice Interface:** Enabling speech-to-text for voters who may have difficulty typing.
3.  **Analytics:** A dashboard for election officials to see what questions are being asked most frequently."

---

### **Slide 11: Conclusion**
**(Speech)**
"In conclusion, Sarika is more than just a chatbot. It is a tool for **digital democracy**.
It empowers voters with accurate information, fights misinformation, and makes the election process more transparent and accessible for everyone.
Thank you for listening. I am happy to take any questions you may have."
