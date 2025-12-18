from pathlib import Path
import json
import os
import shutil
from typing import List, Dict, Any, Optional
from collections import deque
import re
import pickle

import numpy as np

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
 
from langchain_core.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter


class ElectionChatbot:
    def __init__(self, data_path: Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        # No longer using data_path for faqs.json as primary source
        self.kb_path = base_dir / "data" / "knowledge_base"
        self.vector_store_path = base_dir / "data" / "faiss_index"
        self.hybrid_cache_path = base_dir / "data" / "hybrid_cache.pkl"
        
        self._questions_index: List[str] = []
        
        # Caches
        self._cache: Dict[str, str] = {}
        self._cache_order: deque[str] = deque(maxlen=256)
        self._live_cache: Dict[str, str] = {}
        self._definitions: Dict[str, str] = {}
        self._user_aliases: Dict[str, int] = {}
        self._feedback_path = base_dir / "data" / "feedback.jsonl"
        self._last_mtime: float = 0.0

        # Districts
        self._districts: List[str] = [
            "Achham","Arghakhanchi","Baglung","Baitadi","Bajhang","Bajura","Banke","Bara","Bardiya","Bhaktapur","Bhojpur","Chitwan","Dadeldhura","Dailekh","Dang","Darchula","Dhading","Dhankuta","Dolakha","Dolpa","Doti","Gorkha","Gulmi","Humla","Ilam","Jajarkot","Jhapa","Jumla","Kailali","Kalikot","Kanchanpur","Kapilvastu","Kaski","Kathmandu","Kavrepalanchok","Lalitpur","Lamjung","Mahottari","Makwanpur","Manang","Morang","Mugu","Mustang","Myagdi","Nawalparasi East","Nawalparasi West","Nuwakot","Okhaldhunga","Palpa","Panchthar","Parbat","Parsa","Pyuthan","Rautahat","Rolpa","Rukum East","Rukum West","Rupandehi","Salyan","Sankhuwasabha","Saptari","Sarlahi","Sindhuli","Sindhupalchok","Siraha","Solukhumbu","Sunsari","Surkhet","Syangja","Tanahun","Taplejung","Terhathum","Udayapur","Manang","Mustang","Nawalpur","Parasi","Rasuwa","Kavre"
        ]
        self._district_synonyms: Dict[str, str] = {
            "ktm": "Kathmandu",
            "kathmandu": "Kathmandu",
            "lalitpur": "Lalitpur",
            "patan": "Lalitpur",
            "kavre": "Kavrepalanchok",
            "nawalpur": "Nawalparasi East",
            "parasi": "Nawalparasi West",
            "bardia": "Bardiya",
        }
        self._feelings_pos = {
            "admiration","energetic","interested","enthusiastic","amazed","fascinated","joyful","amused","comfortable","fortunate","affectionate","confident","glad","keen","anticipation","connected","happy","motivated","attracted","content","hopeful","proud","brave","courageous","relaxed","thankful","calm","eager","curious","refreshed","thrilled","cheerful","inspired","relieved","trustful","encouraged","satisfied","touched","warm","peaceful"
        }
        self._feelings_neg = {
            "angry","exhausted","jealous","lonely","nervous","anxious","fearful","puzzled","overwhelmed","ashamed","grief","regretful","bored","bitter","guilty","reluctant","restless","confused","concerned","disappointed","hopeless","hesitant","discouraged","hurt","sad","tense","uneasy","disinterested","impatient","tired","upset","disturbed","disgusted","insecure","uncomfortable","embarrassed","irritated","worried","shocked","frustrated","helpless"
        }
        self._needs_map: Dict[str, List[str]] = {
            "connection": [
                "acceptance","affection","appreciation","belonging","cooperation","communication","community","companionship","empathy","intimacy","love","inclusion","consideration","closeness","mutuality","nurturing","respect","safety","security","stability","to know","to understand","trust","warmth"
            ],
            "physical_wellbeing": [
                "movement","nourishment","rest","sleep","health","sexual expression","safety","touch","shelter","water","food"
            ],
            "honesty": [
                "authenticity","integrity","self-respect","truth","clarity","transparency"
            ],
            "play": [
                "joy","humor","fun","celebration"
            ],
            "peace": [
                "beauty","ease","equality","harmony","inspiration","order","calm"
            ],
            "autonomy": [
                "freedom","independence","space","spontaneity","choice"
            ],
            "meaning": [
                "awareness","challenge","competence","contribution","creativity","discovery","efficacy","purpose","growth","learning","participation","to matter","understanding","stimulation","hope"
            ]
        }

        # LangChain components
        self.hf_api_key = os.getenv("HF_API_KEY")
        self.qa_chain = None
        self.vector_store = None
        self.llm = None
        self.prompt = None
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._corpus_texts: List[str] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._bert_embeddings = None
        self.chain = None
        self.retriever = None
        self.fast_mode: bool = True
        self._max_chars: int = 600
        self.kb_only: bool = True
        
        # Initialize
        self._load_feedback()
        self._init_rag()

    def _load_feedback(self):
        if self._feedback_path.exists():
            try:
                with self._feedback_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("helpful") and rec.get("question"):
                                self._user_aliases[rec["question"].lower()] = 0 
                        except: pass
            except: pass

    def _init_rag(self):
        if FAISS is None:
            return
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            if self.vector_store_path.exists():
                try:
                    self.vector_store = FAISS.load_local(str(self.vector_store_path), self.embeddings, allow_dangerous_deserialization=True)
                except Exception as e:
                    print(f"Error loading vector store: {e}. Rebuilding...")
                    self.vector_store = self._build_vector_store()
            else:
                self.vector_store = self._build_vector_store()
            if not self.vector_store:
                print("Warning: Could not build vector store. Check data/knowledge_base.")
                return
            try:
                self._build_hybrid_index()
            except Exception as e:
                print(f"Hybrid Index Error: {e}")
            try:
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            except Exception:
                self.retriever = None
            if self.hf_api_key:
                try:
                    self.llm = HuggingFaceEndpoint(
                        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                        huggingfacehub_api_token=self.hf_api_key,
                        temperature=0.1,
                        max_new_tokens=512,
                        timeout=120
                    )
                    template = "You are a helpful AI assistant for the Nepal Election Commission.\nUse the following context to answer the user's question accurately.\nIf the answer is not in the context, say \"I don't have that information in my knowledge base.\"\nDo not make up information.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
                    self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                    self.chain = {
                        "context": RunnableLambda(lambda x: self._lc_context(x["question"])),
                        "question": RunnableLambda(lambda x: x["question"])
                    } | self.prompt | self.llm | StrOutputParser()
                except Exception as e:
                    print(f"RAG Init Error: {e}")
                    self.llm = None
        except Exception as e:
            print(f"RAG Init Error: {e}")
            self.llm = None

    def _build_vector_store(self):
        if not self.kb_path.exists():
            return None
        
        # Load .txt files
        try:
            loader = DirectoryLoader(str(self.kb_path), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            docs = loader.load()
            if not docs:
                return None
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            vs = FAISS.from_documents(chunks, self.embeddings)
            vs.save_local(str(self.vector_store_path))
            self._corpus_texts = [(c.page_content or "").strip() for c in chunks if (c.page_content or "").strip()]
            return vs
        except Exception as e:
            print(f"Vector Store Build Error: {e}")
            return None

    def rebuild_index(self):
        if self.vector_store_path.exists():
            shutil.rmtree(self.vector_store_path)
        self._init_rag()

    def _load_data(self) -> None:
        pass # faqs.json is removed
    
    def _build_hybrid_index(self) -> None:
        # Try loading from cache
        if self.hybrid_cache_path.exists():
            try:
                with open(self.hybrid_cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                self._corpus_texts = cache_data["corpus_texts"]
                self._tfidf_vectorizer = cache_data["tfidf_vectorizer"]
                self._tfidf_matrix = cache_data["tfidf_matrix"]
                self._doc_embeddings = cache_data["doc_embeddings"]
                self._definitions = cache_data["definitions"]
                self._bert_embeddings = self.embeddings
                return
            except Exception as e:
                print(f"Cache Load Error: {e}")

        if not self._corpus_texts:
            try:
                loader = DirectoryLoader(str(self.kb_path), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                self._corpus_texts = [(c.page_content or "").strip() for c in chunks if (c.page_content or "").strip()]
            except Exception:
                self._corpus_texts = []
        if not self._corpus_texts:
            return
        self._build_definition_index()
        self._tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._corpus_texts)
        try:
            self._bert_embeddings = self.embeddings
            embeds = self._bert_embeddings.embed_documents(self._corpus_texts)
            m = np.array(embeds, dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
            self._doc_embeddings = m / norms
        except Exception:
            self._bert_embeddings = None
            self._doc_embeddings = None
        
        # Save to cache
        try:
            with open(self.hybrid_cache_path, "wb") as f:
                pickle.dump({
                    "corpus_texts": self._corpus_texts,
                    "tfidf_vectorizer": self._tfidf_vectorizer,
                    "tfidf_matrix": self._tfidf_matrix,
                    "doc_embeddings": self._doc_embeddings,
                    "definitions": self._definitions
                }, f)
        except Exception as e:
            print(f"Cache Save Error: {e}")

    def _build_definition_index(self) -> None:
        self._definitions = {}
        if not self._corpus_texts:
            return
        patterns = [
            r'["“](.+?)["”]\s+(?:means|refers to)\s+(.+?)(?:\.)',
            r"'(.+?)'\s+(?:means|refers to)\s+(.+?)(?:\.)",
            r'\b([A-Z][A-Za-z /]{0,80}?)\b\s+(?:means|refers to)\s+(.+?)(?:\.)',
        ]
        for text in self._corpus_texts:
            for raw_line in text.splitlines():
                line = self._strip_enums_line(raw_line.strip())
                if not line:
                    continue
                for pat in patterns:
                    for m in re.finditer(pat, line):
                        term = (m.group(1) or "").strip()
                        if not term:
                            continue
                        key = term.lower()
                        if key in self._definitions:
                            continue
                        definition = m.group(0).strip()
                        if not definition.endswith("."):
                            definition += "."
                        self._definitions[key] = definition
    
    def _hybrid_search(self, query: str, k: int = 3, threshold: float = 0.4) -> List[str]:
        if not self._corpus_texts:
            return []
        scores = None
        if self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            qv = self._tfidf_vectorizer.transform([query])
            tf_scores = cosine_similarity(qv, self._tfidf_matrix).ravel()
            if scores is None:
                scores = tf_scores
            else:
                scores += tf_scores
        if self._bert_embeddings is not None and self._doc_embeddings is not None:
            q_embed = np.array(self._bert_embeddings.embed_query(query), dtype=np.float32)
            qn = q_embed / (np.linalg.norm(q_embed) + 1e-8)
            bert_scores = self._doc_embeddings @ qn
            if scores is None:
                scores = bert_scores
            else:
                # Weighted sum? Or just sum.
                # Since BERT is more semantic, we trust it, but TF-IDF helps with exact keywords.
                scores += bert_scores
        if scores is None:
            return []
        
        # Filter by threshold
        # Note: if both models are active, max score is 2.0. If only one, max is 1.0.
        # We adjust threshold dynamically if needed, but a fixed reasonable one is safer.
        # If both active, 0.4 is very low. Let's assume 0.5.
        
        idxs = np.argsort(scores)[::-1][:k]
        results = []
        for i in idxs:
            if scores[i] < threshold:
                break
            if 0 <= int(i) < len(self._corpus_texts):
                results.append(self._corpus_texts[int(i)])
        return results
    
    def _find_definition(self, question: str) -> str:
        """
        Heuristic to find definitions like '"Commission" means...'
        """
        # Normalize question to extract the term
        q = question.lower().strip()
        q = re.sub(r"\?+$", "", q)
        # Remove common prefixes
        for prefix in ["what is ", "define ", "meaning of ", "who is "]:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
        
        # Remove common suffixes like " means"
        if q.endswith(" means"):
            q = q[:-6].strip()
        
        # If term is too long, it's likely not a definition request
        if len(q.split()) > 4:
            return ""

        term = q
        if term in self._definitions:
            return self._definitions[term]
        # Patterns to look for in corpus
        # We look for:
        # 1. "Term" means ...
        # 2. Term means ...
        # 3. Term is defined as ...
        patterns = [
            r'["\']' + re.escape(term) + r'["\']\s+(?:means|refers to)\s+([^.\n]+(?:\.[^.\n]+)*)',
            r'\b' + re.escape(term) + r'\b\s+(?:means|refers to)\s+([^.\n]+(?:\.[^.\n]+)*)',
            r'\b' + re.escape(term) + r'\b\s+is\s+defined\s+as\s+([^.\n]+(?:\.[^.\n]+)*)',
        ]
        
        # Scan corpus texts (chunks)
        for text in self._corpus_texts:
            t_low = text.lower()
            if term not in t_low:
                continue
            
            # Try to match patterns in the original text (case insensitive search, but return original case)
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    # Found a definition!
                    # Return the full sentence/clause.
                    definition = m.group(0).strip()
                    # Ensure it ends with a dot if the capture didn't include it fully
                    if not definition.endswith("."):
                        definition += "."
                    return definition
        return ""

    def kb_status(self) -> Dict[str, Any]:
        try:
            txt_files = list(self.kb_path.rglob("*.txt"))
        except Exception:
            txt_files = []
        empty = 0
        for p in txt_files:
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
                if not (s or "").strip():
                    empty += 1
            except Exception:
                empty += 1
        return {
            "files_count": len(txt_files),
            "chunks_count": len(self._corpus_texts or []),
            "tfidf_ready": self._tfidf_vectorizer is not None and self._tfidf_matrix is not None,
            "bert_ready": self._doc_embeddings is not None,
            "definitions_count": len(self._definitions or {}),
            "empty_files": empty,
        }

    def _fast_answer(self, question: str) -> str:
        # 1. Try definition lookup for short queries
        defn = self._find_definition(question)
        if defn:
            return defn

        # 2. Hybrid search
        hits = self._hybrid_search(question, k=1, threshold=0.45)
        if hits:
            # Try to extract strictly if possible
            strict = self._extract_from_kb_chunk(question, hits[0])
            if strict:
                return strict
            # Fallback to general extraction
            best = self._extract_best_answer(question, hits[0].strip())
            if best:
                return best
        
        # 3. Fallback to vector store similarity search
        if self.vector_store:
            try:
                # Similarity search with score threshold
                docs_and_scores = self.vector_store.similarity_search_with_score(question, k=1)
                for d, score in docs_and_scores or []:
                    # FAISS L2 distance: lower is better. 
                    # 0 is identical. > 1 is usually bad for normalized vectors.
                    # But if embeddings are not normalized, it varies.
                    # sentence-transformers/all-MiniLM-L6-v2 usually normalized.
                    # A safe threshold for L2 might be around 1.0 or 1.2 depending on data.
                    # Let's assume < 1.0 is relevant.
                    if score < 1.1:
                        t = (d.page_content or "").strip()
                        if t:
                            best = self._extract_best_answer(question, t)
                            if best:
                                return best
            except Exception:
                pass
        return ""
    
    def _lc_context(self, question: str) -> str:
        hybrid = self._hybrid_search(question, k=3)
        parts: List[str] = []
        if hybrid:
            parts.extend(hybrid[:3])
        if self.retriever is not None:
            try:
                docs = self.retriever.invoke(question)
                for d in docs or []:
                    t = (getattr(d, "page_content", "") or "").strip()
                    if t:
                        parts.append(t)
            except Exception:
                pass
        seen = set()
        uniq = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        return "\n\n".join(uniq[:3])

    def get_response(self, user_text: str) -> str:
        data = self.get_answer_and_next_by_question(user_text, "en")
        ans = data.get("answer", "")
        if not ans or "don't have that information" in ans.lower():
            try:
                print(f"FAISS available: {FAISS is not None}", flush=True)
                if FAISS is not None:
                    if not self.vector_store:
                        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        if self.vector_store_path.exists():
                            try:
                                self.vector_store = FAISS.load_local(str(self.vector_store_path), self.embeddings, allow_dangerous_deserialization=True)
                            except Exception:
                                self.vector_store = None
                        if not self.vector_store:
                            self.vector_store = self._build_vector_store()
                    if self.vector_store:
                        docs = self.vector_store.similarity_search(user_text, k=3)
                        if docs:
                            parts: List[str] = []
                            for d in docs:
                                t = (d.page_content or "").strip()
                                if t:
                                    parts.append(t)
                            if parts:
                                return "\n\n".join(parts[:3])
            except Exception:
                pass
        return ans or "I could not generate an answer."

    def get_answer_and_next_by_question(self, question: str, lang: str = "np") -> Dict[str, Any]:
        question = (question or "").strip()
        
        # 1. Topic Gating
        if not self._is_election_en(question):
             return {
                "answer": "I am an election chatbot. Please ask me about voting, registration, or polling in Nepal.",
                "next": [],
                "meta": {"emotion": {"valence": "neutral", "labels": []}, "needs": [], "district": None}
            }
            
        # 2. Check Cache
        key = question.lower()
        if key in self._cache:
            cached_answer = self._cache[key]
            em = self._analyze_emotions_needs(question)
            return {
                "answer": cached_answer,
                "next": self._suggest_next(question, em),
                "meta": {"emotion": em["emotion"], "needs": em["needs"], "district": self._detect_district(question)}
            }

        # 3. Use RAG
        answer = ""
        if self.fast_mode or self.kb_only:
            answer = self._fast_answer(question)
        elif self.chain and not self.kb_only:
            try:
                answer = (self.chain.invoke({"question": question}) or "").strip()
            except Exception as e:
                print(f"RAG Error: {e}", flush=True)
                answer = ""
        
        if not answer or "don't have that information" in answer.lower():
            try:
                hybrid = self._hybrid_search(question, k=3)
                if hybrid:
                    answer = "\n\n".join(hybrid[:3])
                elif self.vector_store:
                    docs = self.vector_store.similarity_search(question, k=3)
                    if docs:
                        parts: List[str] = []
                        for d in docs:
                            txt = (d.page_content or "").strip()
                            if txt:
                                parts.append(txt)
                        if parts:
                            answer = "\n\n".join(parts[:3])
            except Exception:
                pass
            if not answer:
                answer = "I'm sorry, I don't have information on that specific topic yet. Try asking about registration, polling, or voter ID."

        em = self._analyze_emotions_needs(question)
        preface = self._compose_empathy_preface(em)
        cleaned = self._extract_best_answer(question, answer)
        final_answer = f"{preface}{cleaned}" if preface else cleaned
        # Cache the final user-visible answer
        self._cache[key] = final_answer
        return {"answer": final_answer, "next": self._suggest_next(question, em), "meta": {"emotion": em["emotion"], "needs": em["needs"], "district": self._detect_district(question)}}

    def _recommend_next_generic(self, q: str) -> List[str]:
        return []

    def list_questions(self, limit: int = 20) -> List[str]:
        return [
            "How do I register to vote?",
            "Where is my polling center?",
            "What ID do I need?",
            "When are the elections?",
            "Can I vote from abroad?"
        ]

    def _detect_district(self, text: str) -> str | None:
        t = (text or "").lower()
        try:
            toks = set(re.findall(r"[a-z]+", t))
        except:
            toks = set(t.split())
        for alias, canonical in self._district_synonyms.items():
            if alias in toks: return canonical
        for d in self._districts:
            if d.lower() in toks: return d
        return None

    def _is_election_en(self, t: str) -> bool:
        words = [
            "election","vote","voter","register","registration","polling center","polling station","polling place","where to vote",
            "identification","id","citizenship","voter card","hours","time","open","close","update","correction","name change",
            "address","status","check registration","accessible","assistance","disabled","office","ward office","district election office","deo","ecn",
            "lost","lost voter card","lost voter id","replacement","replace voter card","replace id",
            "parliament","house of representatives","national assembly","provincial assembly","local government","municipality","rural municipality","ward",
            "constituency","fptp","first past the post","proportional representation","pr","threshold","quota","gender quota","inclusion",
            "electoral system","candidate eligibility","nomination","campaign","code of conduct","observers","election results","counting",
            "ballot","ballot paper","ballot counting","postal vote","absentee","by-election","by election","re-run","rerun",
            "complaint","complaints","dispute resolution","schedule","election date","government formation","dissolution","prime minister","speaker","chairperson",
            "commission", "secretary", "commissioner", "act", "law", "duty", "duties", "power", "function", "temporary position", "expert service",
            "expense", "expenses", "ceiling", "fine", "punishment", "audit", "auditor", "monitoring",
            "budget", "accounts", "financial", "consultation", "advice", "awareness", "technology", "facilities", "allowance", "insurance", "expenditure",
            "procurement", "procure", "goods", "suggestions", "direction", "pleading", "legal", "gender", "inclusive", "suspension", "remove",
            "departmental action", "appeal", "annual report", "delegation", "liaison", "ministry", "rules", "orders", "directives", "manuals", "repeal",
            "democracy", "multiparty", "political party", "political parties", "civil society", "media", "press", "deposit", "candidate requirements",
            "symbol", "security", "crime", "impersonation", "tendered", "captured", "disturbance", "peace", "punishment",
            "how", "what", "where", "when", "who", "why" 
        ]
        # Check FAQ keywords too
        # Removed hardcoded FAQ keyword check as faqs.json is gone.
        # Future: Could use knowledge base or config for this.
        return any(w in t.lower() for w in words)
    
    def _normalize_tokens(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()
    
    def _analyze_emotions_needs(self, t: str) -> Dict[str, Any]:
        toks = set(self._normalize_tokens(t))
        pos = sorted([w for w in self._feelings_pos if w in toks])
        neg = sorted([w for w in self._feelings_neg if w in toks])
        if neg and not pos:
            valence = "negative"
            labels = neg[:4]
        elif pos and not neg:
            valence = "positive"
            labels = pos[:4]
        elif pos or neg:
            valence = "mixed"
            labels = (neg[:2] + pos[:2])[:4]
        else:
            valence = "neutral"
            labels = []
        needs_hits: Dict[str, int] = {}
        for cat, words in self._needs_map.items():
            hits = sum(1 for w in words if w in toks)
            if hits:
                needs_hits[cat] = hits
        top_needs = [k for k, _ in sorted(needs_hits.items(), key=lambda kv: (-kv[1], kv[0]))][:3]
        return {
            "emotion": {"valence": valence, "labels": labels},
            "needs": top_needs
        }
    
    def _compose_empathy_preface(self, em: Dict[str, Any]) -> str:
        val = em.get("emotion", {}).get("valence")
        labs = em.get("emotion", {}).get("labels", [])
        needs = em.get("needs", [])
        if val in ("negative", "mixed"):
            parts = []
            if labs:
                parts.append(f"I understand you may be feeling {' and '.join(labs)}. ")
            if needs:
                parts.append(f"Let’s focus on {', '.join(needs)} needs. ")
            return "".join(parts)
        return ""
    
    def _strip_enums_line(self, line: str) -> str:
        s = re.sub(r"^\s*(\d+|[•\-*])\s*[.)]?\s*", "", line or "")
        s = re.sub(r"^\s*answer\s*(?:\([a-zA-Z ]+\))?\s*:\s*", "", s, flags=re.IGNORECASE)
        return s.strip()
    
    def _postprocess_answer(self, text: str, question: str) -> str:
        if not text:
            return ""
        lines = [self._strip_enums_line(l) for l in (text.splitlines())]
        cleaned = []
        qlow = (question or "").strip().lower()
        for l in lines:
            if not l:
                continue
            # Filter out Nepali text (Devanagari)
            if any('\u0900' <= c <= '\u097f' for c in l):
                continue
            if "?" in l:
                continue
            if l.lower() == qlow:
                continue
            cleaned.append(l)
        out = "\n".join(cleaned).strip()
        return out
    
    def _extract_best_answer(self, question: str, text: str) -> str:
        base = self._postprocess_answer(text, question)
        if not base:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", base)
        qtok = set(self._normalize_tokens(question))
        best = ""
        best_score = -1.0
        for s in sentences:
            ss = s.strip()
            if not ss or "?" in ss:
                continue
            toks = set(self._normalize_tokens(ss))
            overlap = len(qtok & toks)
            has_def = (" is " in ss.lower()) or (" are " in ss.lower())
            score = overlap + (2.0 if has_def else 0.0) - (0.01 * abs(len(ss) - 140))
            if score > best_score:
                best_score = score
                best = ss
        best = best or base
        best = self._strip_enums_line(best)
        if len(best) > self._max_chars:
            best = best[: self._max_chars].rstrip() + "…"
        return best

    def _extract_from_kb_chunk(self, question: str, chunk: str) -> Optional[str]:
        """
        Given a knowledge-base chunk (text), try to find the user's question line
        and return the paragraph(s) immediately following it as the authoritative answer.
        """
        if not chunk:
            return None
        qnorm = " ".join(self._normalize_tokens(question))
        # Split into lines and look for candidate question lines
        lines = [l.rstrip() for l in chunk.splitlines()]
        # compress lines: consider blank lines as separators
        for i, line in enumerate(lines):
            if not line:
                continue
            # normalize candidate line
            ln = line.lower()
            # skip lines that are clearly answers (no question mark)
            candidate_score = 0
            # direct question markers
            if re.search(r"\bquestion\b[:\s]", ln) or ln.strip().endswith("?") or re.match(r"^\s*\d+\.", ln):
                # compute overlap of tokens
                toks = set(self._normalize_tokens(ln))
                qtokens = set(self._normalize_tokens(question))
                overlap = len(toks & qtokens)
                # also check substring
                if qnorm and qnorm in " ".join(self._normalize_tokens(ln)):
                    candidate_score += 3
                candidate_score += overlap
            if candidate_score >= 1:
                # Extract following paragraph lines until blank or next numbered/question line
                out_lines = []
                for j in range(i+1, len(lines)):
                    l2 = lines[j].strip()
                    if not l2:
                        break
                    if re.match(r"^\s*(\d+|Question)\b", l2) or l2.endswith("?"):
                        break
                    
                    # Filter out Nepali answers or labeled answers that aren't English
                    # Check for "Answer (Nepali):" or similar
                    if re.match(r"^\s*Answer\s*\(Nepali\)\s*:", l2, re.IGNORECASE):
                        continue
                    # Also skip if it looks like just Nepali text (heuristic: Devanagari block)
                    # U+0900 to U+097F is Devanagari
                    if any('\u0900' <= c <= '\u097f' for c in l2):
                        continue
                        
                    out_lines.append(l2)
                
                if out_lines:
                    ans = " ".join(out_lines).strip()
                    ans = self._strip_enums_line(ans)
                    return ans
        return None
    
    def _suggest_next(self, q: str, em: Dict[str, Any]) -> List[str]:
        # User requested to remove next chips
        return []

    def refresh_live(self) -> None:
        """
        Refreshes live data cache (now only from local KB or static info).
        No external scraping.
        """
        # We can keep some static placeholders if needed, or just clear it.
        # For now, just keep the snippet extraction from existing KB if we want,
        # but 'refresh_live' implies fetching new data. 
        # Since we are removing scraping, this might just be a no-op or internal re-index.
        pass

    def record_feedback(self, question: str, helpful: bool) -> None:
        try:
            with self._feedback_path.open("a", encoding="utf-8") as f:
                record = {"question": question, "helpful": helpful, "timestamp": self._last_mtime} # timestamp mock
                f.write(json.dumps(record) + "\n")
        except:
            pass
