# ============================================================
# DocFinder AI — RAG Based AI Assistant
# ZeroGPU-compatible version
# ============================================================

# IMPORTANT:
# spaces MUST be imported before torch / transformers
import spaces

import os
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
# CONFIGURATION
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


# ============================================================
# TESSERACT
# ============================================================

# Windows local environment
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# GLOBAL STATE
# ============================================================

all_chunks: List[Document] = []

vector_store = None

bm25 = None

embedding_model = None

current_filename = ""

# ------------------------------------------------------------
# ZERO GPU / LLM
# ------------------------------------------------------------
#
# IMPORTANT:
# On Hugging Face ZeroGPU, model placement must happen at
# module scope so ZeroGPU can intercept/register it.
#
# Locally, use CPU.
#
# ZeroGPU sets SPACES_ZERO_GPU=1.
# ------------------------------------------------------------

if os.getenv("SPACES_ZERO_GPU") == "1":
    LLM_DEVICE = "cuda"
else:
    LLM_DEVICE = -1


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embeddings():

    global embedding_model

    if embedding_model is None:

        print("Loading embedding model...")

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print("Embedding model loaded.")

    return embedding_model


# ============================================================
# ZERO GPU — LAZY LOAD QWEN
# ============================================================
#
# IMPORTANT:
# Do NOT initialize the CUDA-backed pipeline at import time on
# Hugging Face ZeroGPU. The pipeline is created only after the
# @spaces.GPU function receives GPU resources.
#
# Locally, the same function uses CPU.
# ============================================================

llm = None


# ============================================================
# ZERO GPU — ACTUAL LLM GENERATION
# ============================================================

@spaces.GPU(duration=15)
def run_llm(prompt: str) -> str:

    global llm

    if llm is None:

        print("Preparing Qwen 0.5B...")

        llm = pipeline(
            "text-generation",
            model=LLM_MODEL,
            tokenizer=LLM_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
            device=LLM_DEVICE
        )

        print("Qwen pipeline ready.")

    result = llm(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    if not result:
        return ""

    return result[0].get(
        "generated_text",
        ""
    ).strip()


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf(file_path: str) -> List[Document]:

    documents = []

    pdf = pymupdf.open(file_path)

    print(f"Pages loaded: {len(pdf)}")

    for page_number, page in enumerate(pdf):

        text = page.get_text("text").strip()

        # ----------------------------------------------------
        # OCR fallback for scanned PDFs
        # ----------------------------------------------------

        if not text:

            print(
                f"Page {page_number + 1}: "
                f"No direct text. Running OCR..."
            )

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2)
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            text = pytesseract.image_to_string(
                img
            ).strip()

        if text:

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number + 1
                    }
                )
            )

    pdf.close()

    return documents


def extract_docx(file_path: str) -> List[Document]:

    doc = DocxDocument(file_path)

    text_parts = []

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:
            text_parts.append(text)

    text = "\n".join(text_parts)

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "page": 1
            }
        )
    ]


def extract_txt(file_path: str) -> List[Document]:

    with open(
        file_path,
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
                "page": 1
            }
        )
    ]


def extract_document(file_path: str) -> List[Document]:

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":

        return extract_pdf(file_path)

    if extension == ".docx":

        return extract_docx(file_path)

    if extension == ".txt":

        return extract_txt(file_path)

    raise ValueError(
        "Unsupported file type. "
        "Please upload PDF, DOCX, or TXT."
    )


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
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

    return chunks


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(chunks: List[Document]):

    global bm25

    tokenized_documents = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    if tokenized_documents:

        bm25 = BM25Okapi(
            tokenized_documents
        )

    else:

        bm25 = None


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(file):

    global all_chunks
    global vector_store
    global current_filename

    if file is None:

        return (
            "",
            "Please upload a document."
        )

    try:

        file_path = file

        current_filename = Path(
            file_path
        ).name

        print(
            f"Processing: {current_filename}"
        )

        documents = extract_document(
            file_path
        )

        print(
            f"Extracted documents: "
            f"{len(documents)}"
        )

        if not documents:

            return (
                "",
                "No readable text found."
            )

        all_chunks = create_chunks(
            documents
        )

        print(
            f"Created chunks: "
            f"{len(all_chunks)}"
        )

        embeddings = load_embeddings()

        print(
            f"Creating FAISS index for "
            f"{len(all_chunks)} chunks..."
        )

        vector_store = FAISS.from_documents(
            all_chunks,
            embeddings
        )

        build_bm25(
            all_chunks
        )

        print(
            "Document ready."
        )

        return (
            "ready",
            f"✓ {current_filename} is ready"
        )

    except Exception as e:

        print(
            f"PROCESS ERROR: {repr(e)}"
        )

        return (
            "",
            f"Error: {str(e)}"
        )


# ============================================================
# QUERY INTENT
# ============================================================

def detect_intent(query: str) -> str:

    q = query.lower().strip()

    if any(
        word in q
        for word in [
            "skill",
            "skills",
            "technology",
            "technologies",
            "programming language",
            "tools",
            "technical"
        ]
    ):
        return "skills"

    if any(
        word in q
        for word in [
            "education",
            "degree",
            "college",
            "university",
            "qualification"
        ]
    ):
        return "education"

    if any(
        word in q
        for word in [
            "experience",
            "work experience",
            "employment",
            "worked"
        ]
    ):
        return "experience"

    if any(
        word in q
        for word in [
            "project",
            "projects"
        ]
    ):
        return "projects"

    if any(
        word in q
        for word in [
            "contact",
            "email",
            "phone",
            "mobile"
        ]
    ):
        return "contact"

    return "general"


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query: str) -> List[str]:

    intent = detect_intent(query)

    queries = [query]

    if intent == "skills":

        queries.extend([
            "technical skills",
            "programming languages",
            "technologies",
            "tools",
            "technical expertise"
        ])

    elif intent == "education":

        queries.extend([
            "education",
            "degree",
            "university",
            "college",
            "qualification"
        ])

    elif intent == "experience":

        queries.extend([
            "work experience",
            "professional experience",
            "employment",
            "career"
        ])

    elif intent == "projects":

        queries.extend([
            "projects",
            "project experience",
            "projects developed"
        ])

    return queries


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    query: str,
    text: str
) -> float:

    query_words = set(
        re.findall(
            r"\b\w+\b",
            query.lower()
        )
    )

    text_words = set(
        re.findall(
            r"\b\w+\b",
            text.lower()
        )
    )

    if not query_words:

        return 0.0

    overlap = (
        query_words.intersection(
            text_words
        )
    )

    return len(overlap) / len(query_words)


# ============================================================
# INTENT SCORE
# ============================================================

def intent_score(
    query: str,
    text: str
) -> float:

    intent = detect_intent(query)

    text_lower = text.lower()

    if intent == "skills":

        terms = [
            "technical skills",
            "programming languages",
            "technologies",
            "tools",
            "skills",
            "python",
            "sql",
            "mongodb"
        ]

    elif intent == "education":

        terms = [
            "education",
            "degree",
            "university",
            "college",
            "qualification"
        ]

    elif intent == "experience":

        terms = [
            "experience",
            "employment",
            "worked",
            "professional"
        ]

    elif intent == "projects":

        terms = [
            "project",
            "projects",
            "developed"
        ]

    elif intent == "contact":

        terms = [
            "email",
            "phone",
            "mobile",
            "contact"
        ]

    else:

        terms = []

    if not terms:

        return 0.0

    matches = sum(
        1
        for term in terms
        if term in text_lower
    )

    return matches / len(terms)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    k: int = FINAL_TOP_K
) -> List[Document]:

    if vector_store is None:

        return []

    embeddings = load_embeddings()

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    semantic_results = []

    expanded_queries = expand_query(
        query
    )

    for expanded_query in expanded_queries:

        try:

            results = vector_store.similarity_search(
                expanded_query,
                k=SEMANTIC_TOP_K
            )

            semantic_results.extend(
                results
            )

        except Exception as e:

            print(
                f"Semantic search error: "
                f"{repr(e)}"
            )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_results = []

    if bm25 is not None:

        query_tokens = query.lower().split()

        scores = bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:BM25_TOP_K]

        bm25_results = [
            all_chunks[i]
            for i in ranked_indices
        ]

    # --------------------------------------------------------
    # Unique candidate documents
    # --------------------------------------------------------

    candidates = []

    seen = set()

    for doc in (
        semantic_results +
        bm25_results
    ):

        text = doc.page_content

        if text not in seen:

            seen.add(text)

            candidates.append(doc)

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    scored = []

    semantic_texts = [
        doc.page_content
        for doc in semantic_results
    ]

    for doc in candidates:

        text = doc.page_content

        semantic = 0.0

        if text in semantic_texts:

            semantic = 1.0

        bm25_value = 0.0

        if bm25 is not None:

            try:

                index = all_chunks.index(
                    doc
                )

                query_tokens = (
                    query.lower().split()
                )

                scores = bm25.get_scores(
                    query_tokens
                )

                maximum = max(scores)

                if maximum > 0:

                    bm25_value = (
                        scores[index] /
                        maximum
                    )

            except Exception:

                bm25_value = 0.0

        keyword = keyword_score(
            query,
            text
        )

        intent = intent_score(
            query,
            text
        )

        score = (
            semantic * 1.0
            + bm25_value * 1.5
            + keyword * 2.0
            + intent * 5.0
        )

        scored.append(
            (
                score,
                doc
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if scored:

        print(
            f"RETRIEVAL SCORE: "
            f"{scored[0][0]:.3f}"
        )

        print(
            scored[0][1].page_content[
                :500
            ]
        )

    final_documents = [
        item[1]
        for item in scored[:k]
    ]

    print(
        f"FINAL RETRIEVED CHUNKS: "
        f"{len(final_documents)}"
    )

    return final_documents


# ============================================================
# BROAD QUERY DETECTION
# ============================================================

def is_broad_query(
    query: str
) -> bool:

    q = query.lower().strip()

    broad_phrases = [
        "tell me about this document",
        "tell me about the document",
        "summarize this document",
        "summarize the document",
        "give me a summary",
        "what is this document about",
        "what does this document contain",
        "overview",
        "summarize"
    ]

    return any(
        phrase in q
        for phrase in broad_phrases
    )


# ============================================================
# OVERVIEW RETRIEVAL
# ============================================================

def get_overview_documents():

    if not all_chunks:

        return []

    return all_chunks[:FINAL_TOP_K]


# ============================================================
# DIRECT SECTION EXTRACTION
# ============================================================

def extract_section_answer(
    query: str
) -> str:

    intent = detect_intent(
        query
    )

    if intent not in [
        "skills",
        "education",
        "experience",
        "projects",
        "contact"
    ]:

        return ""

    heading_map = {

        "skills": [
            "technical skills",
            "skills",
            "technical expertise",
            "programming languages",
            "technologies",
            "tools",
            "tools and technologies"
        ],

        "education": [
            "education",
            "academic background",
            "qualification"
        ],

        "experience": [
            "experience",
            "work experience",
            "professional experience"
        ],

        "projects": [
            "projects",
            "project experience"
        ],

        "contact": [
            "contact",
            "contact information"
        ]
    }

    headings = heading_map[
        intent
    ]

    for chunk in all_chunks:

        lines = (
            chunk.page_content
            .splitlines()
        )

        for i, line in enumerate(lines):

            normalized = (
                line.strip()
                .lower()
                .rstrip(":")
            )

            if normalized in headings:

                collected = []

                # ------------------------------------------------
                # Inline content after heading
                # ------------------------------------------------

                original = line.strip()

                if ":" in original:

                    remainder = (
                        original.split(
                            ":",
                            1
                        )[1].strip()
                    )

                    if remainder:

                        collected.append(
                            remainder
                        )

                # ------------------------------------------------
                # Following lines
                # ------------------------------------------------

                for next_line in lines[
                    i + 1:
                ]:

                    clean = next_line.strip()

                    if not clean:

                        continue

                    lower = clean.lower()

                    # Stop at another section heading
                    if (
                        clean.isupper()
                        and len(clean) < 80
                    ):

                        break

                    if any(
                        lower.startswith(
                            heading + ":"
                        )
                        for heading in [
                            "education",
                            "experience",
                            "projects",
                            "certifications",
                            "achievements",
                            "contact",
                            "summary",
                            "objective"
                        ]
                    ):

                        break

                    collected.append(
                        clean
                    )

                if collected:

                    answer = "\n".join(
                        collected
                    ).strip()

                    if answer:

                        return answer

    return ""


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    query: str,
    documents: List[Document]
) -> str:

    # --------------------------------------------------------
    # Direct section answer
    # --------------------------------------------------------

    direct_answer = (
        extract_section_answer(
            query
        )
    )

    if direct_answer:

        return direct_answer

    # --------------------------------------------------------
    # No retrieved information
    # --------------------------------------------------------

    if not documents:

        return (
            "I couldn't find that "
            "information in the uploaded "
            "document."
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for doc in documents:

        context_parts.append(
            doc.page_content
        )

    context = "\n\n".join(
        context_parts
    )

    # Keep prompt reasonably small
    context = context[:6000]

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are DocFinder AI, a document question-answering assistant.

Answer the user's question using ONLY the information in the
uploaded document context below.

Do not use outside knowledge.

If the answer is not present in the context, say:

I couldn't find that information in the uploaded document.

Be concise and directly answer the question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:

        result = run_llm(
            prompt
        )

        answer = result

        if not answer:

            return (
                "I couldn't find that "
                "information in the uploaded "
                "document."
            )

        return answer.strip()

    except Exception as e:

        print(
            f"LLM ERROR: {repr(e)}"
        )

        return (
            f"Error: {str(e)}"
        )


# ============================================================
# SOURCE CREATION
# ============================================================

def create_sources(
    documents: List[Document]
) -> str:

    if not documents:

        return ""

    cards = []

    for index, doc in enumerate(
        documents,
        start=1
    ):

        page = doc.metadata.get(
            "page",
            "N/A"
        )

        preview = (
            doc.page_content
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 220:

            preview = (
                preview[:220] +
                "..."
            )

        cards.append(
            f"""
            <div class="source-card">
                <div class="source-number">
                    Source {index}
                </div>

                <div class="source-page">
                    Page {page}
                </div>

                <div class="source-preview">
                    {preview}
                </div>
            </div>
            """
        )

    return f"""
    <div class="sources-wrapper">

        <div class="sources-title">
            Sources
        </div>

        {''.join(cards)}

    </div>
    """


# ============================================================
# SEARCH LOADER
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
# SEARCH FUNCTION
# ============================================================

def search_document(
    query: str
):

    if not query or not query.strip():

        yield (
            "",
            "",
            "",
            "",
            gr.update(
                interactive=True
            )
        )

        return

    # --------------------------------------------------------
    # SHOW LOADER IMMEDIATELY
    # --------------------------------------------------------

    yield (
        gr.update(
            value=LOADER_HTML,
            visible=True
        ),
        "",
        "",
        "",
        gr.update(
            interactive=False
        )
    )

    try:

        query = query.strip()

        print(
            f"\nQUESTION: {query}"
        )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        if is_broad_query(
            query
        ):

            documents = (
                get_overview_documents()
            )

        else:

            documents = retrieve_documents(
                query
            )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = generate_answer(
            query,
            documents
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = create_sources(
            documents
        )

        print(
            "Answer generated."
        )

        # ----------------------------------------------------
        # HIDE LOADER
        # ----------------------------------------------------

        yield (
            gr.update(
                value="",
                visible=False
            ),
            answer,
            sources,
            "",
            gr.update(
                interactive=True
            )
        )

    except Exception as e:

        print(
            f"SEARCH ERROR: {repr(e)}"
        )

        yield (
            gr.update(
                value="",
                visible=False
            ),
            f"Error: {str(e)}",
            "",
            "",
            gr.update(
                interactive=True
            )
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
        gr.update(
            visible=True
        ),
        gr.update(
            visible=False
        ),
        None,
        "",
        "",
        "",
        ""
    )


# ============================================================
# CSS
# ============================================================

CSS = """

/* =========================================================
   GLOBAL
   ========================================================= */

body {

    background:
        radial-gradient(
            circle at top left,
            #20245a 0%,
            #0b1028 35%,
            #050816 75%
        ) !important;

    color: #ffffff !important;

}

.gradio-container {

    max-width: 1200px !important;

    margin: auto !important;

    background:
        radial-gradient(
            circle at 20% 0%,
            rgba(91, 77, 255, 0.16),
            transparent 30%
        ),
        radial-gradient(
            circle at 80% 10%,
            rgba(0, 188, 255, 0.10),
            transparent 30%
        ),
        #070b1c !important;

}


/* =========================================================
   HEADER
   ========================================================= */

.hero {

    text-align: center;

    padding: 50px 20px 25px;

}

.hero-title {

    font-size: 46px;

    font-weight: 800;

    letter-spacing: -1px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #a78bfa,
            #67e8f9
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

}

.hero-subtitle {

    margin-top: 10px;

    color: #aab4d0;

    font-size: 17px;

}


/* =========================================================
   GLASS CARD
   ========================================================= */

.glass-card {

    background:
        rgba(
            17,
            24,
            55,
            0.68
        ) !important;

    border:

        1px solid

        rgba(
            255,
            255,
            255,
            0.09
        ) !important;

    border-radius: 24px !important;

    box-shadow:
        0 20px 70px
        rgba(0,0,0,0.30);

    padding: 28px !important;

}


/* =========================================================
   UPLOAD
   ========================================================= */

.upload-title {

    font-size: 25px;

    font-weight: 700;

    margin-bottom: 8px;

}

.upload-description {

    color: #9ba8c7;

    margin-bottom: 20px;

}


/* =========================================================
   FEATURES
   ========================================================= */

.feature-card {

    padding: 22px;

    border-radius: 18px;

    background:
        rgba(
            255,
            255,
            255,
            0.035
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.07
        );

    min-height: 110px;

}

.feature-title {

    font-weight: 700;

    margin-bottom: 8px;

}

.feature-text {

    color: #9aa7c5;

    font-size: 14px;

}


/* =========================================================
   TEXTBOX
   ========================================================= */

textarea,
input {

    background:
        rgba(
            7,
            12,
            30,
            0.75
        ) !important;

    border:
        1px solid
        rgba(
            126,
            102,
            255,
            0.30
        ) !important;

    color: white !important;

    border-radius: 16px !important;

}


/* =========================================================
   ANSWER
   ========================================================= */

.answer-card {

    background:
        rgba(
            17,
            24,
            55,
            0.72
        );

    border:
        1px solid
        rgba(
            123,
            97,
            255,
            0.24
        );

    border-radius: 22px;

    padding: 25px;

    margin-top: 20px;

    line-height: 1.7;

}


/* =========================================================
   SOURCES
   ========================================================= */

.sources-wrapper {

    margin-top: 22px;

}

.sources-title {

    font-size: 21px;

    font-weight: 700;

    margin-bottom: 14px;

}

.source-card {

    background:
        rgba(
            255,
            255,
            255,
            0.035
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.07
        );

    border-radius: 16px;

    padding: 17px;

    margin-bottom: 12px;

}

.source-number {

    color: #a78bfa;

    font-weight: 700;

}

.source-page {

    color: #67e8f9;

    font-size: 13px;

    margin-top: 4px;

}

.source-preview {

    color: #aab4d0;

    margin-top: 9px;

    font-size: 14px;

    line-height: 1.5;

}


/* =========================================================
   CUSTOM LOADER
   ========================================================= */

.loader-area {

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    padding: 55px 20px;

}

.circle-loader {

    position: relative;

    width: 80px;

    height: 80px;

}

.circle-loader span {

    position: absolute;

    left: 50%;

    top: 50%;

    width: 8px;

    height: 8px;

    margin-left: -4px;

    margin-top: -4px;

    border-radius: 50%;

    background: #8b7cff;

    transform:
        rotate(var(--angle))
        translateY(-32px);

    animation:
        loaderFade 1.2s
        linear infinite;

    animation-delay:
        calc(
            var(--angle) / 360 * -1.2s
        );

}

@keyframes loaderFade {

    0% {

        opacity: 0.20;

    }

    50% {

        opacity: 1;

    }

    100% {

        opacity: 0.20;

    }

}

.loader-text {

    margin-top: 22px;

    color: #b9c4df;

    font-size: 15px;

}


/* =========================================================
   BUTTON
   ========================================================= */

button {

    border-radius: 14px !important;

}


/* =========================================================
   MOBILE
   ========================================================= */

@media (
    max-width: 700px
) {

    .hero-title {

        font-size: 34px;

    }

    .glass-card {

        padding: 18px !important;

    }

}

"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title=APP_TITLE
) as demo:

    # --------------------------------------------------------
    # PAGE 1 — UPLOAD
    # --------------------------------------------------------

    upload_page = gr.Column(
        visible=True
    )

    with upload_page:

        gr.HTML(
            """
            <div class="hero">

                <div class="hero-title">
                    DocFinder AI
                </div>

                <div class="hero-subtitle">
                    Your intelligent document
                    question-answering assistant
                </div>

            </div>
            """
        )

        with gr.Column(
            elem_classes="glass-card"
        ):

            gr.HTML(
                """
                <div class="upload-title">
                    Upload your document
                </div>

                <div class="upload-description">
                    Upload a PDF, DOCX, or TXT
                    file and ask questions using
                    semantic search.
                </div>
                """
            )

            file_input = gr.File(
                label="Document",
                file_types=[
                    ".pdf",
                    ".docx",
                    ".txt"
                ],
                type="filepath"
            )

            process_button = gr.Button(
                "Process Document",
                variant="primary"
            )

            process_status = gr.Markdown(
                ""
            )

            process_loader = gr.HTML(
                "",
                visible=False
            )

            with gr.Row():

                gr.HTML(
                    """
                    <div class="feature-card">

                        <div class="feature-title">
                            🔎 Semantic Search
                        </div>

                        <div class="feature-text">
                            Understand questions
                            even when the exact
                            words are different.
                        </div>

                    </div>
                    """
                )

                gr.HTML(
                    """
                    <div class="feature-card">

                        <div class="feature-title">
                            📄 Document Grounded
                        </div>

                        <div class="feature-text">
                            Answers are generated
                            from your uploaded
                            document.
                        </div>

                    </div>
                    """
                )

                gr.HTML(
                    """
                    <div class="feature-card">

                        <div class="feature-title">
                            ⚡ Fast Retrieval
                        </div>

                        <div class="feature-text">
                            FAISS vector search
                            combined with lexical
                            retrieval.
                        </div>

                    </div>
                    """
                )

    # --------------------------------------------------------
    # PAGE 2 — SEARCH
    # --------------------------------------------------------

    search_page = gr.Column(
        visible=False
    )

    with search_page:

        gr.HTML(
            """
            <div class="hero">

                <div class="hero-title">
                    DocFinder AI
                </div>

                <div class="hero-subtitle">
                    Ask anything about your
                    uploaded document
                </div>

            </div>
            """
        )

        with gr.Column(
            elem_classes="glass-card"
        ):

            question = gr.Textbox(
                label="Ask your question",
                placeholder=(
                    "Ask a question and press Enter..."
                ),
                lines=2
            )

            search_loader = gr.HTML(
                "",
                visible=True
            )

            answer_output = gr.Markdown(
                "",
                elem_classes="answer-card"
            )

            sources_output = gr.HTML(
                ""
            )

            search_status = gr.Markdown(
                ""
            )

            reset_button = gr.Button(
                "↻ Try Another Document"
            )


# ============================================================
# PROCESS FLOW
# ============================================================

def process_with_loader(file):

    # Show the CUSTOM loader first. The previous version only
    # changed visibility, so the loader component was visible but
    # contained no HTML.
    yield (
        gr.update(
            value=PROCESS_LOADER_HTML,
            visible=True
        ),
        "",
        gr.update(
            visible=True
        ),
        gr.update(
            visible=False
        ),
        gr.update(
            visible=False
        )
    )

    status, message = process_document(
        file
    )

    if status == "ready":

        yield (
            gr.update(
                visible=False
            ),
            message,
            gr.update(
                visible=False
            ),
            gr.update(
                visible=True
            ),
            gr.update(
                visible=False
            )
        )

    else:

        yield (
            gr.update(
                visible=False
            ),
            message,
            gr.update(
                visible=False
            ),
            gr.update(
                visible=True
            ),
            gr.update(
                visible=False
            )
        )


# ============================================================
# EVENTS
# ============================================================

process_button.click(

    fn=process_with_loader,

    inputs=[
        file_input
    ],

    outputs=[
        process_loader,
        process_status,
        upload_page,
        search_page,
        search_loader
    ],

    show_progress="hidden"
)


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

    show_progress="hidden",

    js="() => { document.activeElement?.blur(); }"
)


reset_button.click(

    fn=reset_app,

    inputs=[],

    outputs=[
        upload_page,
        search_page,
        file_input,
        process_status,
        search_loader,
        answer_output,
        sources_output
    ],

    show_progress="hidden"
)


# ============================================================
# LAUNCH
# ============================================================

# Required for generator-based custom loader updates to stream
# to the browser immediately.
demo.queue()

demo.launch(
    inbrowser=True,
    css=CSS
)