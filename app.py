import os
import spaces
import re
from pathlib import Path
from typing import List, Dict

import gradio as gr
import pymupdf
import pytesseract

from PIL import Image
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi
from transformers import pipeline


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "DocFinder AI"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SEMANTIC_TOP_K = 8
BM25_TOP_K = 8
FINAL_TOP_K = 4

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

MAX_NEW_TOKENS = 70

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# GLOBAL STATE
# ============================================================

all_chunks: List[Document] = []

vector_store = None
bm25 = None

embedding_model = None
llm = None

current_filename = ""


# ============================================================
# LOADER
# ============================================================

LOADER_HTML = """
<div class="loader-area">

    <div class="circle-loader">

        <span style="--angle:0deg;"></span>
        <span style="--angle:45deg;"></span>
        <span style="--angle:90deg;"></span>
        <span style="--angle:135deg;"></span>
        <span style="--angle:180deg;"></span>
        <span style="--angle:225deg;"></span>
        <span style="--angle:270deg;"></span>
        <span style="--angle:315deg;"></span>

    </div>

    <div class="loader-text">
        Searching your document...
    </div>

</div>
"""


PROCESS_LOADER_HTML = """
<div class="loader-area">

    <div class="circle-loader">

        <span style="--angle:0deg;"></span>
        <span style="--angle:45deg;"></span>
        <span style="--angle:90deg;"></span>
        <span style="--angle:135deg;"></span>
        <span style="--angle:180deg;"></span>
        <span style="--angle:225deg;"></span>
        <span style="--angle:270deg;"></span>
        <span style="--angle:315deg;"></span>

    </div>

    <div class="loader-text">
        Processing your document...
    </div>

</div>
"""


# ============================================================
# CSS
# ============================================================

CSS = """

/* ==========================================================
   GLOBAL
   ========================================================== */

body {
    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(55, 65, 120, 0.14),
            transparent 38%
        ),
        #0b0b0d !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 24px 28px !important;
    background: transparent !important;
}


/* ==========================================================
   HEADER
   ========================================================== */

.app-header {
    text-align: center;
    margin-bottom: 28px;
}

.app-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #91b2ff;
    margin: 0;
}

.app-subtitle {
    margin-top: 8px;
    font-size: 18px;
    color: #9aa9ca;
}


/* ==========================================================
   MAIN CARD
   ========================================================== */

.glass-card {
    background:
        linear-gradient(
            145deg,
            rgba(19, 25, 50, 0.97),
            rgba(13, 18, 38, 0.97)
        );

    border: 1px solid rgba(91, 125, 230, 0.38);

    border-radius: 30px;

    padding: 50px;

    box-shadow:
        0 25px 70px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.025);

    min-height: 600px;
}


/* ==========================================================
   HEADINGS
   ========================================================== */

.section-title {
    color: #f2f4ff;
    font-size: 29px;
    font-weight: 800;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #9caacd;
    font-size: 17px;
    margin-bottom: 30px;
}


/* ==========================================================
   QUESTION BOX
   ========================================================== */

.question-box textarea,
.question-box input {
    background: #29292b !important;

    border: 2px solid #45464d !important;

    border-radius: 10px !important;

    color: #f2f4ff !important;

    font-size: 17px !important;

    padding: 15px !important;

    box-shadow: none !important;
}

.question-box textarea:focus,
.question-box input:focus {
    border-color: #5d75bd !important;

    box-shadow:
        0 0 0 1px
        rgba(93, 117, 189, 0.2) !important;
}

.enter-hint {
    color: #94a4c9;
    font-size: 14px;
    margin-top: 8px;
}


/* ==========================================================
   ANSWER
   ========================================================== */

.answer-card {
    background:
        rgba(15, 25, 60, 0.82);

    border:
        1px solid
        rgba(79, 123, 232, 0.55);

    border-radius: 24px;

    padding: 28px;

    margin-top: 28px;

    color: #eef2ff;

    line-height: 1.75;

    font-size: 18px;
}

.answer-heading {
    color: #8fb1ff;

    font-size: 19px;

    font-weight: 800;

    margin-bottom: 14px;
}

.answer-text {
    color: #e9edff;

    white-space: pre-wrap;
}


/* ==========================================================
   SOURCES
   ========================================================== */

.sources-title {
    color: #f0f2ff;

    font-size: 17px;

    font-weight: 800;

    margin-top: 34px;

    margin-bottom: 12px;
}

.source-card {
    background:
        rgba(29, 34, 55, 0.80);

    border:
        1px solid
        rgba(96, 110, 150, 0.32);

    border-radius: 14px;

    padding: 14px 16px;

    margin-bottom: 10px;
}

.source-number {
    color: #8fb1ff;

    font-weight: 800;
}

.source-text {
    color: #b9c3dd;

    font-size: 14px;

    line-height: 1.55;
}


/* ==========================================================
   STATUS
   ========================================================== */

.status-text {
    text-align: center;

    color: #8faeff;

    font-size: 14px;

    margin-top: 16px;
}


/* ==========================================================
   8-DOT LOADER
   ========================================================== */

.loader-area {
    width: 100%;

    height: 155px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    margin: 5px 0;
}


/*
   IMPORTANT:
   The dots are positioned using rotate + translateY.
   The animation changes ONLY opacity.
   Therefore the dots will no longer collapse
   into the center.
*/

.circle-loader {
    width: 82px;

    height: 82px;

    position: relative;

    animation:
        loader-rotate
        1.8s
        linear
        infinite;
}

.circle-loader span {
    position: absolute;

    width: 13px;

    height: 13px;

    border-radius: 50%;

    background: #9a9a9a;

    left: 34.5px;

    top: 34.5px;

    transform:
        rotate(var(--angle))
        translateY(-32px);

    transform-origin:
        6.5px 6.5px;

    opacity: 0.25;

    animation:
        loader-fade
        1.4s
        ease-in-out
        infinite;
}

.circle-loader span:nth-child(1) {
    animation-delay: 0s;
}

.circle-loader span:nth-child(2) {
    animation-delay: 0.175s;
}

.circle-loader span:nth-child(3) {
    animation-delay: 0.35s;
}

.circle-loader span:nth-child(4) {
    animation-delay: 0.525s;
}

.circle-loader span:nth-child(5) {
    animation-delay: 0.70s;
}

.circle-loader span:nth-child(6) {
    animation-delay: 0.875s;
}

.circle-loader span:nth-child(7) {
    animation-delay: 1.05s;
}

.circle-loader span:nth-child(8) {
    animation-delay: 1.225s;
}


@keyframes loader-fade {

    0% {
        opacity: 0.18;
    }

    50% {
        opacity: 1;
    }

    100% {
        opacity: 0.18;
    }
}


@keyframes loader-rotate {

    from {
        transform: rotate(0deg);
    }

    to {
        transform: rotate(360deg);
    }
}


.loader-text {
    color: #9a9a9a;

    font-size: 14px;

    margin-top: 14px;
}


/* ==========================================================
   FEATURES
   ========================================================== */

.feature-grid {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 15px;

    margin-top: 30px;
}

.feature-card {
    background:
        rgba(26, 32, 55, 0.7);

    border:
        1px solid
        rgba(88, 108, 160, 0.25);

    border-radius: 18px;

    padding: 20px;

    text-align: center;
}

.feature-title {
    color: #dfe6ff;

    font-weight: 700;

    margin-bottom: 6px;
}

.feature-text {
    color: #8f9dbc;

    font-size: 13px;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.primary-button {
    border-radius: 16px !important;

    min-height: 52px !important;

    font-weight: 700 !important;
}

.reset-button {
    width: 100%;

    min-height: 54px !important;

    margin-top: 34px;

    border-radius: 18px !important;

    background:
        rgba(35, 40, 60, 0.72) !important;

    border:
        1px solid
        rgba(100, 112, 150, 0.35) !important;

    color: #eef1ff !important;

    font-size: 18px !important;

    font-weight: 700 !important;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 900px) {

    .gradio-container {
        padding: 16px !important;
    }

    .glass-card {
        padding: 30px 24px;

        border-radius: 24px;
    }

    .app-title {
        font-size: 34px;
    }

    .feature-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 600px) {

    .gradio-container {
        padding: 10px !important;
    }

    .glass-card {
        padding: 22px 16px;

        border-radius: 20px;
    }

    .app-title {
        font-size: 28px;
    }

    .app-subtitle {
        font-size: 15px;
    }

    .section-title {
        font-size: 24px;
    }

    .answer-card {
        padding: 20px;

        font-size: 16px;
    }
}

@media (max-width: 380px) {

    .app-title {
        font-size: 24px;
    }

    .glass-card {
        padding: 18px 12px;
    }
}

"""


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(path: str) -> List[Document]:

    documents = []

    pdf = pymupdf.open(path)

    for page_number, page in enumerate(pdf):

        text = page.get_text("text").strip()

        # Normal text PDF
        if len(text) >= 30:

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": Path(path).name,
                        "page": page_number + 1,
                        "method": "text"
                    }
                )
            )

        else:

            # OCR for scanned pages
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(
                    1.7,
                    1.7
                ),
                alpha=False
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            ocr_text = (
                pytesseract
                .image_to_string(img)
                .strip()
            )

            if ocr_text:

                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "source": Path(path).name,
                            "page": page_number + 1,
                            "method": "ocr"
                        }
                    )
                )

    pdf.close()

    return documents


def extract_docx(path: str) -> List[Document]:

    doc = DocxDocument(path)

    paragraphs = []

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    text = "\n".join(paragraphs)

    if not text:

        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": Path(path).name,
                "page": 1,
                "method": "docx"
            }
        )
    ]


def extract_txt(path: str) -> List[Document]:

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read().strip()

    if not text:

        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": Path(path).name,
                "page": 1,
                "method": "text"
            }
        )
    ]


def extract_document(path: str) -> List[Document]:

    extension = Path(path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".docx":
        return extract_docx(path)

    if extension == ".txt":
        return extract_txt(path)

    raise ValueError(
        "Unsupported file type. "
        "Use PDF, DOCX or TXT."
    )


# ============================================================
# CHUNKING
# ============================================================

def split_documents(
    documents: List[Document]
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            ": ",
            ", ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    cleaned = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if len(text) >= 25:

            cleaned.append(
                Document(
                    page_content=text,
                    metadata=chunk.metadata
                )
            )

    return cleaned


# ============================================================
# EMBEDDINGS
# ============================================================

def load_embeddings():

    global embedding_model

    if embedding_model is None:

        print("Loading embedding model...")

        embedding_model = (
            HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,

                model_kwargs={
                    "device": "cpu"
                },

                encode_kwargs={
                    "normalize_embeddings": True
                }
            )
        )

        print("Embedding model loaded.")

    return embedding_model


def build_faiss(
    chunks: List[Document]
):

    embeddings = load_embeddings()

    print(
        f"Creating FAISS index "
        f"for {len(chunks)} chunks..."
    )

    return FAISS.from_documents(
        chunks,
        embeddings
    )


# ============================================================
# BM25
# ============================================================

def tokenize(text: str) -> List[str]:

    return re.findall(
        r"\b[a-zA-Z0-9+#.\-]+\b",
        text.lower()
    )


def build_bm25(
    chunks: List[Document]
):

    corpus = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    return BM25Okapi(corpus)


def bm25_search(
    query: str,
    k: int = BM25_TOP_K
) -> List[Document]:

    if bm25 is None:
        return []

    scores = bm25.get_scores(
        tokenize(query)
    )

    indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in indices[:k]:

        if scores[index] <= 0:
            continue

        results.append(
            all_chunks[index]
        )

    return results


# ============================================================
# LLM
# ============================================================

def load_llm():

    global llm

    if llm is None:

        print("Loading Qwen 0.5B...")

        llm = pipeline(
            "text-generation",

            model=LLM_MODEL,

            tokenizer=LLM_MODEL,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            return_full_text=False,

            device=-1
        )

        print("Qwen loaded.")

    return llm


# ============================================================
# INTENT
# ============================================================

def normalize_query(
    query: str
) -> str:

    query = query.lower().strip()

    return re.sub(
        r"\s+",
        " ",
        query
    )


def detect_intent(
    query: str
) -> str:

    q = normalize_query(query)

    if any(
        x in q
        for x in [
            "skill",
            "skills",
            "technical skill",
            "technical skills",
            "programming language",
            "programming languages",
            "technology",
            "technologies",
            "tech stack",
            "tools"
        ]
    ):
        return "skills"

    if any(
        x in q
        for x in [
            "education",
            "degree",
            "college",
            "university",
            "qualification",
            "academic"
        ]
    ):
        return "education"

    if any(
        x in q
        for x in [
            "experience",
            "work experience",
            "professional experience",
            "employment",
            "job"
        ]
    ):
        return "experience"

    if any(
        x in q
        for x in [
            "project",
            "projects",
            "developed",
            "built",
            "application"
        ]
    ):
        return "projects"

    if any(
        x in q
        for x in [
            "email",
            "phone",
            "mobile",
            "linkedin",
            "github",
            "contact"
        ]
    ):
        return "contact"

    return "general"


def expand_query(
    query: str
) -> List[str]:

    intent = detect_intent(query)

    q = normalize_query(query)

    variants = [q]

    if intent == "skills":

        variants += [
            "skills",
            "technical skills",
            "technical expertise",
            "programming languages",
            "technologies",
            "tools and technologies",
            "technical skills programming languages technologies"
        ]

    elif intent == "education":

        variants += [
            "education",
            "educational qualification",
            "degree",
            "university",
            "college"
        ]

    elif intent == "experience":

        variants += [
            "work experience",
            "professional experience",
            "employment",
            "job experience"
        ]

    elif intent == "projects":

        variants += [
            "projects",
            "technical projects",
            "projects developed",
            "applications developed"
        ]

    elif intent == "contact":

        variants += [
            "contact information",
            "email",
            "phone",
            "linkedin",
            "github"
        ]

    # Remove duplicates
    return list(dict.fromkeys(variants))


# ============================================================
# INTENT TERMS
# ============================================================

INTENT_TERMS = {

    "skills": [
        "skills",
        "technical skills",
        "programming",
        "programming languages",
        "technologies",
        "technology",
        "tools",
        "frameworks",
        "databases",
        "technical expertise"
    ],

    "education": [
        "education",
        "degree",
        "university",
        "college",
        "qualification",
        "academic"
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "company",
        "role"
    ],

    "projects": [
        "project",
        "projects",
        "developed",
        "built",
        "application"
    ],

    "contact": [
        "email",
        "phone",
        "mobile",
        "linkedin",
        "github",
        "contact"
    ],

    "general": []
}


def keyword_score(
    text: str,
    terms: List[str]
) -> float:

    text_lower = text.lower()

    score = 0.0

    for term in terms:

        if term.lower() in text_lower:

            if len(term.split()) > 1:
                score += 4.0
            else:
                score += 2.0

    return score


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    k: int = SEMANTIC_TOP_K
) -> List[Document]:

    if vector_store is None:
        return []

    return vector_store.similarity_search(
        query,
        k=k
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str
) -> List[Document]:

    if not all_chunks:
        return []

    intent = detect_intent(query)

    variants = expand_query(query)

    candidates: Dict[str, Dict] = {}

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    for variant in variants:

        results = semantic_search(
            variant,
            SEMANTIC_TOP_K
        )

        for rank, doc in enumerate(results):

            key = doc.page_content.strip()

            if key not in candidates:

                candidates[key] = {
                    "doc": doc,
                    "semantic": 0.0,
                    "bm25": 0.0,
                    "keyword": 0.0,
                    "intent": 0.0
                }

            candidates[key]["semantic"] += (
                1.0 / (rank + 1)
            )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for variant in variants:

        results = bm25_search(
            variant,
            BM25_TOP_K
        )

        for rank, doc in enumerate(results):

            key = doc.page_content.strip()

            if key not in candidates:

                candidates[key] = {
                    "doc": doc,
                    "semantic": 0.0,
                    "bm25": 0.0,
                    "keyword": 0.0,
                    "intent": 0.0
                }

            candidates[key]["bm25"] += (
                1.0 / (rank + 1)
            )

    # --------------------------------------------------------
    # Intent scoring
    # --------------------------------------------------------

    intent_terms = INTENT_TERMS.get(
        intent,
        []
    )

    query_terms = tokenize(query)

    for item in candidates.values():

        text = item["doc"].page_content

        item["keyword"] = keyword_score(
            text,
            query_terms
        )

        item["intent"] = keyword_score(
            text,
            intent_terms
        )

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    ranked = []

    for item in candidates.values():

        score = (
            item["semantic"] * 1.0
            +
            item["bm25"] * 1.5
            +
            item["keyword"] * 2.0
            +
            item["intent"] * 5.0
        )

        ranked.append(
            (
                score,
                item["doc"]
            )
        )

    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    final_docs = []

    seen = set()

    for score, doc in ranked:

        key = doc.page_content.strip()

        if key in seen:
            continue

        seen.add(key)

        final_docs.append(doc)

        print(
            f"\nRETRIEVAL SCORE: {score:.3f}"
        )

        print(
            doc.page_content[:500]
        )

        if len(final_docs) >= FINAL_TOP_K:
            break

    return final_docs


# ============================================================
# DIRECT ANSWER EXTRACTION
#
# This is the important fix for questions such as:
# "what are the skills?"
# ============================================================

def extract_section_answer(
    query: str,
    documents: List[Document]
) -> str | None:

    intent = detect_intent(query)

    if intent == "general":
        return None

    if not documents:
        return None

    if intent == "skills":

        heading_patterns = [
            r"technical\s+skills?\s*:?\s*(.*)",
            r"skills?\s*:?\s*(.*)",
            r"technical\s+expertise\s*:?\s*(.*)",
            r"programming\s+languages?\s*:?\s*(.*)",
            r"technologies\s*:?\s*(.*)",
            r"tools\s*(?:and\s+technologies)?\s*:?\s*(.*)"
        ]

        collected = []

        for doc in documents:

            text = doc.page_content.strip()

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            for i, line in enumerate(lines):

                line_clean = re.sub(
                    r"\s+",
                    " ",
                    line
                ).strip()

                # --------------------------------------------
                # Inline section
                # Example:
                # Technical Skills: Python, SQL, MongoDB
                # --------------------------------------------

                for pattern in heading_patterns:

                    match = re.search(
                        pattern,
                        line_clean,
                        flags=re.IGNORECASE
                    )

                    if match:

                        remainder = (
                            match.group(1)
                            .strip()
                        )

                        if remainder:

                            collected.append(
                                remainder
                            )

                        # Capture following list lines
                        for next_line in lines[
                            i + 1:i + 9
                        ]:

                            lower_next = (
                                next_line.lower()
                            )

                            # Stop when another obvious
                            # section begins.
                            if re.match(
                                r"^(education|"
                                r"experience|"
                                r"projects?|"
                                r"certifications?|"
                                r"summary|"
                                r"objective|"
                                r"contact)\b",
                                lower_next
                            ):
                                break

                            collected.append(
                                next_line
                            )

                        break

        # Remove duplicates
        cleaned = []

        for item in collected:

            item = item.strip()

            if (
                item
                and item not in cleaned
            ):
                cleaned.append(item)

        if cleaned:

            return "\n".join(
                cleaned[:12]
            )

    # --------------------------------------------------------
    # Other simple sections
    # --------------------------------------------------------

    if intent == "education":

        patterns = [
            r"education\s*:?\s*(.*)",
            r"qualification\s*:?\s*(.*)",
            r"academic\s*:?\s*(.*)"
        ]

        for doc in documents:

            for line in doc.page_content.splitlines():

                line = line.strip()

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        line,
                        flags=re.IGNORECASE
                    )

                    if match and match.group(1).strip():

                        return match.group(1).strip()

    return None


# ============================================================
# BROAD QUERY
# ============================================================

def is_broad_query(
    query: str
) -> bool:

    q = normalize_query(query)

    patterns = [
        "tell me about this document",
        "tell me about the document",
        "summarize this document",
        "summarize the document",
        "summary of this document",
        "what is this document about",
        "give me an overview",
        "overview of the document"
    ]

    return any(
        p in q
        for p in patterns
    )


def get_overview_documents():

    if not all_chunks:
        return []

    if len(all_chunks) <= FINAL_TOP_K:
        return all_chunks

    indices = [
        0,
        len(all_chunks) // 3,
        (2 * len(all_chunks)) // 3,
        len(all_chunks) - 1
    ]

    result = []

    seen = set()

    for index in indices:

        doc = all_chunks[index]

        key = doc.page_content.strip()

        if key not in seen:

            result.append(doc)

            seen.add(key)

    return result[:FINAL_TOP_K]


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    documents: List[Document]
) -> str:

    parts = []

    for i, doc in enumerate(
        documents,
        1
    ):

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        parts.append(
            f"""
SOURCE {i}
Page: {page}

{doc.page_content}
""".strip()
        )

    return (
        "\n\n-------------------------\n\n"
    ).join(parts)


# ============================================================
# LLM ANSWER
# ============================================================

@spaces.GPU
def generate_answer(
    query: str,
    documents: List[Document]
) -> str:

    if not documents:

        return (
            "I couldn't find that information "
            "in the uploaded document."
        )

    # --------------------------------------------------------
    # First attempt deterministic extraction
    # --------------------------------------------------------

    direct_answer = extract_section_answer(
        query,
        documents
    )

    if direct_answer:

        return direct_answer

    # --------------------------------------------------------
    # Otherwise use LLM
    # --------------------------------------------------------

    context = build_context(
        documents
    )

    prompt = f"""
You are DocFinder AI.

Answer the question using ONLY the
document context below.

Question:
{query}

Document context:
{context}

Rules:

- Do not use outside knowledge.
- Do not guess.
- Do not invent information.
- Do not infer a skill from a project.
- Do not infer a skill from GitHub.
- Do not infer information from a job title.
- If the answer is not explicitly present,
  say exactly:

"I couldn't find that information in the uploaded document."

Give a concise answer.

Answer:
"""

    model = load_llm()

    result = model(prompt)

    if not result:

        return (
            "I couldn't find that information "
            "in the uploaded document."
        )

    answer = result[0].get(
        "generated_text",
        ""
    ).strip()

    answer = re.sub(
        r"^(answer|response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    if not answer:

        return (
            "I couldn't find that information "
            "in the uploaded document."
        )

    return answer

# ============================================================
# SOURCES
# ============================================================

def create_sources(
    documents: List[Document]
) -> str:

    if not documents:
        return ""

    html = ""

    for index, doc in enumerate(
        documents,
        1
    ):

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        text = doc.page_content.strip()

        preview = text[:300]

        if len(text) > 300:
            preview += "..."

        preview = (
            preview
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        html += f"""
        <div class="source-card">

            <div>
                <span class="source-number">
                    Source {index}
                </span>

                &nbsp; • &nbsp;

                Page {page}
            </div>

            <div class="source-text">
                {preview}
            </div>

        </div>
        """

    return html


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(file):

    global all_chunks
    global vector_store
    global bm25
    global current_filename

    if file is None:

        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "⚠️ Please upload a document first."
        )

        return

    # SHOW LOADER IMMEDIATELY
    yield (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(
            value=PROCESS_LOADER_HTML,
            visible=True
        ),
        ""
    )

    try:

        current_filename = Path(
            file.name
        ).name

        print(
            f"\nProcessing: {current_filename}"
        )

        # Extract
        documents = extract_document(
            file.name
        )

        if not documents:

            yield (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                "❌ No readable text found."
            )

            return

        print(
            f"Extracted documents: "
            f"{len(documents)}"
        )

        # Split
        all_chunks = split_documents(
            documents
        )

        print(
            f"Created chunks: "
            f"{len(all_chunks)}"
        )

        if not all_chunks:

            yield (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                "❌ Could not create document chunks."
            )

            return

        # FAISS
        vector_store = build_faiss(
            all_chunks
        )

        # BM25
        bm25 = build_bm25(
            all_chunks
        )

        # Load LLM once
        load_llm()

        print("Document ready.")

        # Go to search page
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )

    except Exception as e:

        print(
            "PROCESSING ERROR:",
            repr(e)
        )

        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            f"❌ Error: {str(e)}"
        )


# ============================================================
# SEARCH
# ============================================================

def search_document(query):

    if not query or not query.strip():

        yield (
            gr.update(visible=False),
            "",
            "",
            "",
            gr.update(interactive=True)
        )

        return

    query = query.strip()

    # ========================================================
    # FIRST YIELD
    #
    # Loader appears immediately.
    # Textbox becomes disabled.
    # Therefore cursor disappears.
    # ========================================================

    yield (
        gr.update(
            value=LOADER_HTML,
            visible=True
        ),
        "",
        "",
        "",
        gr.update(interactive=False)
    )

    try:

        print(
            f"\nQUESTION: {query}"
        )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        if is_broad_query(query):

            retrieved_docs = (
                get_overview_documents()
            )

        else:

            retrieved_docs = (
                retrieve_documents(query)
            )

        print(
            f"\nFINAL RETRIEVED CHUNKS: "
            f"{len(retrieved_docs)}"
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        answer = generate_answer(
            query,
            retrieved_docs
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = create_sources(
            retrieved_docs
        )

        answer_html = f"""
        <div class="answer-card">

            <div class="answer-heading">
                ✦ ANSWER
            </div>

            <div class="answer-text">
                {answer}
            </div>

        </div>
        """

        sources_html = f"""
        <div class="sources-title">
            Sources
        </div>

        {sources}
        """

        # ====================================================
        # FINAL
        #
        # No "Answer generated..." text.
        # Cursor/input enabled again.
        # ====================================================

        yield (
            gr.update(
                value=LOADER_HTML,
                visible=False
            ),
            answer_html,
            sources_html,
            "",
            gr.update(interactive=True)
        )

    except Exception as e:

        print(
            "SEARCH ERROR:",
            repr(e)
        )

        yield (
            gr.update(
                value=LOADER_HTML,
                visible=False
            ),
            "",
            "",
            f"❌ Error: {str(e)}",
            gr.update(interactive=True)
        )


# ============================================================
# RESET
# ============================================================

def reset_app():

    global all_chunks
    global vector_store
    global bm25
    global current_filename

    all_chunks = []

    vector_store = None

    bm25 = None

    current_filename = ""

    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        gr.update(
            value=PROCESS_LOADER_HTML,
            visible=False
        ),
        "",
        "",
        "",
        "",
        gr.update(interactive=True)
    )


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title=APP_TITLE
) as app:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="app-header">

            <div class="app-title">
                ◈ DocFinder AI
            </div>

            <div class="app-subtitle">
                RAG Based AI Assistant for Private Documents
            </div>

        </div>
        """
    )


    # ========================================================
    # UPLOAD PAGE
    # ========================================================

    with gr.Column(
        visible=True,
        elem_classes=["glass-card"]
    ) as upload_page:

        gr.HTML(
            """
            <div class="section-title">
                Upload your document
            </div>

            <div class="section-subtitle">
                Upload a private document and ask questions naturally.
            </div>
            """
        )

        file_upload = gr.File(
            label="",
            show_label=False,
            file_types=[
                ".pdf",
                ".docx",
                ".txt"
            ],
            type="filepath"
        )

        process_button = gr.Button(
            "Process Document",
            variant="primary",
            elem_classes=[
                "primary-button"
            ]
        )

        process_loader = gr.HTML(
            PROCESS_LOADER_HTML,
            visible=False
        )

        process_status = gr.HTML(
            "",
            elem_classes=[
                "status-text"
            ]
        )

        gr.HTML(
            """
            <div class="feature-grid">

                <div class="feature-card">

                    <div class="feature-title">
                        🔒 Private
                    </div>

                    <div class="feature-text">
                        Your document is processed locally.
                    </div>

                </div>

                <div class="feature-card">

                    <div class="feature-title">
                        🔎 Semantic Search
                    </div>

                    <div class="feature-text">
                        Ask questions naturally.
                    </div>

                </div>

                <div class="feature-card">

                    <div class="feature-title">
                        ⚡ RAG Powered
                    </div>

                    <div class="feature-text">
                        Answers are grounded in your document.
                    </div>

                </div>

            </div>
            """
        )


    # ========================================================
    # SEARCH PAGE
    # ========================================================

    with gr.Column(
        visible=False,
        elem_classes=["glass-card"]
    ) as search_page:

        gr.HTML(
            """
            <div class="section-title">
                Search your document
            </div>

            <div class="section-subtitle">
                Ask questions naturally.
            </div>
            """
        )

        question = gr.Textbox(
            label="",
            show_label=False,
            placeholder="Ask something about your document...",
            lines=1,
            elem_classes=[
                "question-box"
            ],
            autofocus=True
        )

        gr.HTML(
            """
            <div class="enter-hint">
                Press Enter to search
            </div>
            """
        )

        # Custom loader
        search_loader = gr.HTML(
            LOADER_HTML,
            visible=False
        )

        answer_output = gr.HTML(
            ""
        )

        # Intentionally empty
        search_status = gr.HTML(
            "",
            elem_classes=[
                "status-text"
            ]
        )

        sources_output = gr.HTML(
            ""
        )

        reset_button = gr.Button(
            "↻ Try Another Document",
            elem_classes=[
                "reset-button"
            ]
        )


    # ========================================================
    # EVENTS
    # ========================================================

    process_button.click(
        fn=process_document,
        inputs=[
            file_upload
        ],
        outputs=[
            upload_page,
            search_page,
            process_loader,
            process_status
        ],
        show_progress="hidden"
    )


    # ========================================================
    # ENTER TO SEARCH
    # ========================================================

    question.submit(
        fn=search_document,
        inputs=[
            question
        ],
        outputs=[
            search_loader,
            answer_output,
            sources_output,
            search_status,
            question
        ],
        show_progress="hidden"
    )


    # ========================================================
    # RESET
    # ========================================================

    reset_button.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            upload_page,
            search_page,
            file_upload,
            process_loader,
            process_status,
            answer_output,
            sources_output,
            search_status,
            question
        ],
        show_progress="hidden"
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    app.launch(
        inbrowser=True,
        css=CSS
    )