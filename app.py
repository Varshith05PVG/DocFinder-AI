import os
import gradio as gr
import traceback
import requests

# DOCUMENT PROCESSING IMPORTS
# ============================================================
import fitz
import pytesseract
import shutil
from PIL import Image
from docx import Document

# ============================================================
# LANGCHAIN / RAG IMPORTS
# ============================================================

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import (
    Document as LangChainDocument
)

from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_community.vectorstores import (
    FAISS
)

from rank_bm25 import BM25Okapi

# ============================================================
# LLM IMPORT
# ============================================================
# ============================================================
# APP CONFIGURATION
# ============================================================

APP_TITLE = "DocFinder AI"

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

LLM_MODEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

SEMANTIC_TOP_K = 8
BM25_TOP_K = 8
FINAL_TOP_K = 4

MAX_NEW_TOKENS = 70


# ============================================================
# GLOBAL VARIABLES
# ============================================================

embedding_model = None

vector_store = None

bm25 = None

llm = None

current_filename = None

all_chunks = []


# ============================================================
# LOADER HTML
# ============================================================

PROCESS_LOADER_HTML = """
<div class="loader">
    ✦ Processing your document...
</div>
"""

SEARCH_LOADER_HTML = """
<div class="loader">
    ✦ Searching your document...
</div>
"""


# ============================================================
# CUSTOM CSS
# ============================================================

CSS = """
:root {
    --bg: #070b14;
    --card: rgba(15, 22, 38, 0.78);
    --card-border: rgba(120, 140, 180, 0.18);
    --text: #f4f7ff;
    --muted: #9aa7bd;
    --accent: #7c5cff;
    --accent-2: #00d9ff;
}


/* ----------------------------------------------------------
   BODY
---------------------------------------------------------- */

body {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124, 92, 255, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(0, 217, 255, 0.12),
            transparent 30%
        ),
        var(--bg);
}


/* ----------------------------------------------------------
   MAIN GRADIO CONTAINER
---------------------------------------------------------- */

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: transparent !important;
}


/* ----------------------------------------------------------
   MAIN WRAPPER
---------------------------------------------------------- */

#main-wrapper {
    padding: 35px 20px 50px 20px;
}


/* ----------------------------------------------------------
   HERO
---------------------------------------------------------- */

.hero {
    text-align: center;
    padding: 35px 20px 25px 20px;
}


.hero-title {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #9c8cff,
        #00d9ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.hero-subtitle {
    color: var(--muted);
    font-size: 18px;
    margin-bottom: 5px;
}


.hero-description {
    color: #78869e;
    font-size: 14px;
}


/* ----------------------------------------------------------
   FEATURE CARDS
---------------------------------------------------------- */

.feature-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 22px !important;
    min-height: 120px;
    backdrop-filter: blur(18px);
    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);
}


.feature-title {
    color: var(--text);
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}


.feature-description {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}


/* ----------------------------------------------------------
   GLASS CARDS
---------------------------------------------------------- */

.glass-card {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 22px !important;
    padding: 25px !important;
    backdrop-filter: blur(18px);
    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.28);
}


/* ----------------------------------------------------------
   SECTION HEADINGS
---------------------------------------------------------- */

.section-title {
    color: var(--text);
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 5px;
}


.section-description {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}


/* ----------------------------------------------------------
   PRIMARY BUTTON
---------------------------------------------------------- */

.primary-btn {
    border: none !important;
    border-radius: 12px !important;

    background: linear-gradient(
        135deg,
        var(--accent),
        #5d7cff
    ) !important;

    color: white !important;
    font-weight: 700 !important;

    box-shadow:
        0 8px 25px rgba(124, 92, 255, 0.25);
}


.primary-btn:hover {
    transform: translateY(-1px);

    box-shadow:
        0 12px 30px rgba(124, 92, 255, 0.35);
}


/* ----------------------------------------------------------
   SECONDARY BUTTON
---------------------------------------------------------- */

.secondary-btn {
    border-radius: 12px !important;
}


/* ----------------------------------------------------------
   STATUS
---------------------------------------------------------- */

.status {
    color: var(--muted);
    text-align: center;
}


/* ----------------------------------------------------------
   LOADER
---------------------------------------------------------- */

.loader {
    text-align: center;
    color: #9c8cff;
    font-weight: 600;
    padding: 10px;
}


/* ----------------------------------------------------------
   ANSWER / SOURCES
---------------------------------------------------------- */

.answer-box {
    border-radius: 16px !important;
}


.sources-box {
    border-radius: 16px !important;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer {
    text-align: center;
    color: #5f6c82;
    font-size: 12px;
    padding-top: 30px;
}
"""



# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_path):
    """
    Extract text from a PDF.

    Normal text extraction is attempted first.

    If a page contains little/no extractable text,
    OCR is used as a fallback.
    """

    text_parts = []

    # --------------------------------------------------------
    # TESSERACT CONFIGURATION
    # --------------------------------------------------------

    detected_tesseract = shutil.which("tesseract")

    if detected_tesseract:
        pytesseract.pytesseract.tesseract_cmd = detected_tesseract

    # --------------------------------------------------------
    # OPEN PDF
    # --------------------------------------------------------

    document = fitz.open(file_path)

    try:

        for page_number, page in enumerate(document):

            # ------------------------------------------------
            # NORMAL PDF TEXT EXTRACTION
            # ------------------------------------------------

            text = page.get_text("text")

            if text and text.strip():

                text_parts.append(
                    text.strip()
                )

                continue

            # ------------------------------------------------
            # OCR FALLBACK
            # ------------------------------------------------

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            try:

                ocr_text = pytesseract.image_to_string(
                    image
                )

                if ocr_text and ocr_text.strip():

                    text_parts.append(
                        ocr_text.strip()
                    )

            except Exception as exc:

                raise RuntimeError(
                    f"OCR failed on PDF page "
                    f"{page_number + 1}: {exc}"
                )

    finally:

        document.close()

    return "\n\n".join(
        text_parts
    ).strip()


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_path):
    """
    Extract text from a DOCX file
    (paragraphs + tables).
    """

    document = Document(file_path)

    parts = []

    # --------------------------------------------------------
    # PARAGRAPHS
    # --------------------------------------------------------

    for paragraph in document.paragraphs:

        if paragraph.text and paragraph.text.strip():

            parts.append(
                paragraph.text.strip()
            )

    # --------------------------------------------------------
    # TABLES
    # --------------------------------------------------------

    for table in document.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text and cell.text.strip()
            ]

            if cells:

                parts.append(
                    " | ".join(cells)
                )

    return "\n\n".join(
        parts
    ).strip()


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file_path):
    """
    Extract text from a TXT file.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1"
    ]

    for encoding in encodings:

        try:

            with open(
                file_path,
                "r",
                encoding=encoding
            ) as file:

                return file.read().strip()

        except UnicodeDecodeError:

            continue

    raise RuntimeError(
        "Could not decode the TXT file."
    )


# ============================================================
# DOCUMENT EXTRACTION ROUTER
# ============================================================

def extract_document(file_path):
    """
    Detect the uploaded file type and
    call the appropriate extractor.
    """

    if not file_path:

        raise ValueError(
            "No document was selected."
        )

    extension = (
        os.path.splitext(file_path)[1]
        .lower()
    )

    if extension == ".pdf":

        return extract_pdf(
            file_path
        )

    elif extension == ".docx":

        return extract_docx(
            file_path
        )

    elif extension == ".txt":

        return extract_txt(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported file type. "
            "Please upload PDF, DOCX, or TXT."
        )


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text):
    """
    Split extracted document text into
    overlapping chunks.
    """

    if not text or not text.strip():

        return []

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                ""
            ],
        )
    )

    chunks = splitter.split_text(
        text
    )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# FAISS VECTOR STORE
# ============================================================

def create_vector_store(chunks):
    """
    Create a FAISS vector store
    from document chunks.
    """

    global embedding_model
    global vector_store

    if not chunks:

        raise ValueError(
            "No document chunks available."
        )

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------
    if embedding_model is None:
        embedding_model = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    # --------------------------------------------------------
    # CREATE LANGCHAIN DOCUMENTS
    # --------------------------------------------------------

    documents = []

    for index, chunk in enumerate(chunks):

        documents.append(
            LangChainDocument(
                page_content=chunk,
                metadata={
                    "chunk_id": index
                }
            )
        )

    # --------------------------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------------------------

    vector_store = (
        FAISS.from_documents(
            documents,
            embedding_model
        )
    )

    return vector_store


# ============================================================
# BM25 INDEX
# ============================================================

def create_bm25_index(chunks):
    """
    Create a BM25 keyword-search index
    from document chunks.
    """

    global bm25

    if not chunks:

        raise ValueError(
            "No document chunks available."
        )

    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    return bm25


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    top_k=SEMANTIC_TOP_K
):
    """
    Retrieve chunks using FAISS
    semantic similarity.
    """

    if vector_store is None:

        return []

    documents = (
        vector_store.similarity_search(
            query,
            k=top_k
        )
    )

    return documents


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    query,
    top_k=BM25_TOP_K
):
    """
    Retrieve chunks using BM25
    keyword matching.
    """

    if bm25 is None or not all_chunks:

        return []

    query_tokens = (
        query.lower().split()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            all_chunks[index]
        )

    return results


# ============================================================
# COMBINED RETRIEVAL
# ============================================================

def retrieve_documents(query):
    """
    Retrieve relevant document chunks using:

    1. FAISS semantic similarity
    2. BM25 keyword matching

    Results are merged and duplicates are removed.
    """

    if not query or not query.strip():

        return []

    semantic_documents = (
        semantic_search(
            query,
            SEMANTIC_TOP_K
        )
    )

    keyword_documents = (
        keyword_search(
            query,
            BM25_TOP_K
        )
    )

    results = []

    seen = set()

    # --------------------------------------------------------
    # SEMANTIC RESULTS
    # --------------------------------------------------------

    for document in semantic_documents:

        content = (
            document.page_content.strip()
        )

        if content and content not in seen:

            seen.add(content)

            results.append(
                content
            )

    # --------------------------------------------------------
    # KEYWORD RESULTS
    # --------------------------------------------------------

    for content in keyword_documents:

        content = content.strip()

        if content and content not in seen:

            seen.add(content)

            results.append(
                content
            )

    # --------------------------------------------------------
    # FINAL TOP K
    # --------------------------------------------------------

    return results[:FINAL_TOP_K]


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(file_path):
    """
    Current document processing pipeline:

    Upload
       ↓
    Detect file type
       ↓
    Extract text
       ↓
    Create chunks
       ↓
    Create embeddings
       ↓
    Build FAISS index
       ↓
    Build BM25 index
    """

    global current_filename
    global all_chunks

    if not file_path:

        return (
            "Please upload a PDF, DOCX, or TXT file."
        )

    try:

        # ----------------------------------------------------
        # FILE INFORMATION
        # ----------------------------------------------------

        current_filename = os.path.basename(
            file_path
        )

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        text = extract_document(
            file_path
        )

        if not text:

            return (
                "### ⚠ No readable text found\n\n"
                f"`{current_filename}` does not "
                "contain readable text."
            )

        # ----------------------------------------------------
        # CREATE CHUNKS
        # ----------------------------------------------------

        chunks = create_chunks(
            text
        )

        if not chunks:

            return (
                "### ⚠ Could not create document chunks\n\n"
                f"`{current_filename}` did not "
                "produce usable content."
            )

        # ----------------------------------------------------
        # STORE CHUNKS
        # ----------------------------------------------------

        all_chunks = chunks

        # ----------------------------------------------------
        # CREATE FAISS VECTOR STORE
        # ----------------------------------------------------

        create_vector_store(
            chunks
        )

        # ----------------------------------------------------
        # CREATE BM25 INDEX
        # ----------------------------------------------------

        create_bm25_index(
            chunks
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        word_count = len(
            text.split()
        )

        character_count = len(
            text
        )

        chunk_count = len(
            chunks
        )

        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        return (
            "### ✓ Document processed successfully\n\n"
            f"**File:** `{current_filename}`  \n"
            f"**Words:** `{word_count:,}`  \n"
            f"**Characters:** `{character_count:,}`  \n"
            f"**Sections prepared:** `{chunk_count:,}`  \n"
            f"**Search index:** Ready\n\n"
            "Your document is ready for questions."
        )

    except Exception as exc:

        return (
            "### ⚠ Processing failed\n\n"
            f"`{str(exc)}`"
        )


# ============================================================
# ANSWER GENERATION
# ============================================================
def generate_answer(question, retrieved_chunks):
    """
    Generate a high-quality answer using Google Gemini's free API.
    """
    if not retrieved_chunks:
        return "No relevant information was found in the uploaded document."

    context = "\n\n".join(retrieved_chunks)

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        return (
            f"**Most relevant passage:**\n\n"
            f"> {retrieved_chunks[0].strip()}"
        )

    prompt = f"""You are a document question-answering assistant.

Answer the user's question using ONLY the information in the context below.

If the answer is not present in the context, say:
"I could not find that information in the document."

Do not invent facts. Be concise and direct.

Document context:
{context}

Question: {question}

Answer:"""

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 300
                }
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return answer

    except Exception as exc:
        return (
            f"**AI answer failed, showing most relevant passage instead:**\n\n"
            f"> {retrieved_chunks[0].strip()}\n\n"
            f"*(Error: {str(exc)[:150]})*"
        )
# ============================================================
# SOURCE FORMATTER
# ============================================================

def create_sources(
    retrieved_chunks
):
    """
    Format retrieved document chunks
    as readable sources.
    """

    if not retrieved_chunks:

        return "No sources available."

    source_text = (
        "### Retrieved Sources\n\n"
    )

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        source_text += (
            f"**Source {index}**\n\n"
            f"{chunk}\n\n"
            "---\n\n"
        )

    return source_text


# ============================================================
# QUESTION FUNCTION
# ============================================================

def ask_question(question):
    """
    Retrieve relevant document chunks,
    generate an answer using Qwen,
    and display the retrieved sources.
    """

    if not question or not question.strip():
        return (
            "",
            "Please enter a question."
        )

    if vector_store is None:
        return (
            "",
            "Please upload and process a document first."
        )

    try:

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        retrieved_chunks = retrieve_documents(
            question
        )

        if not retrieved_chunks:
            return (
                "No relevant information was found "
                "in the uploaded document.",
                "No sources available."
            )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        # ----------------------------------------------------
        # CREATE SOURCES
        # ----------------------------------------------------

        sources = create_sources(
            retrieved_chunks
        )

        return (
            answer,
            sources
        )

    except Exception as exc:
        print("========== EXCEPTION ==========")
        print("TYPE:", type(exc))
        print("ARGS:", exc.args)
        print("REPR:", repr(exc))
        import traceback
        traceback.print_exc()
        print("===============================")

        return (
            "",
            "Answer generation failed. Check Container Logs."
        )
# ============================================================
# PROCESS LOADER
# ============================================================

def process_with_loader(
    file_path
):
    """
    Show the processing loader immediately,
    then process the document.
    """

    yield (
        gr.update(
            value=PROCESS_LOADER_HTML,
            visible=True
        ),
        ""
    )

    try:

        result = process_document(
            file_path
        )

        yield (
            gr.update(
                value="",
                visible=False
            ),
            result
        )

    except Exception as exc:

        yield (
            gr.update(
                value="",
                visible=False
            ),
            f"Processing failed: `{str(exc)}`"
        )


# ============================================================
# QUESTION LOADER
# ============================================================

def ask_with_loader(
    question
):
    """
    Show the search loader while
    retrieval and answer generation run.
    """

    yield (
        gr.update(
            value=SEARCH_LOADER_HTML,
            visible=True
        ),
        "",
        "",
        "Searching your document..."
    )

    try:

        answer, sources = ask_question(
            question
        )

        yield (
            gr.update(
                value="",
                visible=False
            ),
            answer,
            sources,
            ""
        )

    except Exception as exc:

        yield (
            gr.update(
                value="",
                visible=False
            ),
            "",
            "",
            f"Search failed: `{str(exc)}`"
        )


# ============================================================
# RESET
# ============================================================

def reset_app():
    """
    Reset document state and UI.
    """

    global embedding_model
    global vector_store
    global bm25
    global llm
    global current_filename
    global all_chunks

    embedding_model = None
    vector_store = None
    bm25 = None
    llm = None

    current_filename = None
    all_chunks = []

    return (
        None,
        "",
        "",
        "",
        ""
    )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title=APP_TITLE,
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate"
    )
) as demo:

    # ========================================================
    # MAIN WRAPPER
    # ========================================================

    with gr.Column(
        elem_id="main-wrapper"
    ):

        # ====================================================
        # HERO
        # ====================================================

        gr.HTML(
            """
            <div class="hero">

                <div class="hero-title">
                    ✦ DOCFINDER AI
                </div>

                <div class="hero-subtitle">
                    Ask questions. Find answers.
                    From your documents.
                </div>

                <div class="hero-description">
                    Upload your document and interact
                    with its content using natural language.
                </div>

            </div>
            """
        )

        # ====================================================
        # FEATURE CARDS
        # ====================================================

        with gr.Row():

            with gr.Column(
                scale=1,
                elem_classes=["feature-card"]
            ):

                gr.HTML(
                    """
                    <div class="feature-title">
                        🔒 Private & Secure
                    </div>

                    <div class="feature-description">
                        Your documents stay within the
                        application workflow.
                    </div>
                    """
                )

            with gr.Column(
                scale=1,
                elem_classes=["feature-card"]
            ):

                gr.HTML(
                    """
                    <div class="feature-title">
                        📄 Multiple File Formats
                    </div>

                    <div class="feature-description">
                        Upload PDF, DOCX, and TXT documents.
                    </div>
                    """
                )

            with gr.Column(
                scale=1,
                elem_classes=["feature-card"]
            ):

                gr.HTML(
                    """
                    <div class="feature-title">
                        💬 Document Q&A
                    </div>

                    <div class="feature-description">
                        Ask questions and get answers
                        from your uploaded documents.
                    </div>
                    """
                )

        # ====================================================
        # UPLOAD SECTION
        # ====================================================

        with gr.Column(
            elem_classes=["glass-card"]
        ):

            gr.HTML(
                """
                <div class="section-title">
                    Upload Your Document
                </div>

                <div class="section-description">
                    Upload a PDF, DOCX, or TXT file to
                    get started.
                </div>
                """
            )

            file_input = gr.File(
                label="Choose your document",
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
                elem_classes=["primary-btn"]
            )

            process_loader = gr.HTML(
                value="",
                visible=False
            )

            process_status = gr.Markdown(
                "",
                elem_classes=["status"]
            )

        # ====================================================
        # QUESTION SECTION
        # ====================================================

        with gr.Column(
            elem_classes=["glass-card"]
        ):

            gr.HTML(
                """
                <div class="section-title">
                    Ask Your Document
                </div>

                <div class="section-description">
                    Ask a question about the document
                    you uploaded.
                </div>
                """
            )

            question = gr.Textbox(
                label="",
                placeholder=(
                    "Ask a question about your document..."
                ),
                lines=3
            )

            ask_button = gr.Button(
                "Ask Question",
                variant="primary",
                elem_classes=["primary-btn"]
            )

            search_loader = gr.HTML(
                value="",
                visible=False
            )

            search_status = gr.Markdown(
                "",
                elem_classes=["status"]
            )

            gr.Markdown(
                "### Answer"
            )

            answer_output = gr.Markdown(
                "Your answer will appear here.",
                elem_classes=["answer-box"]
            )

            gr.Markdown(
                "### Sources"
            )

            sources_output = gr.Markdown(
                "Sources will appear here.",
                elem_classes=["sources-box"]
            )

        # ====================================================
        # RESET BUTTON
        # ====================================================

        reset_button = gr.Button(
            "↻  Upload Another Document",
            variant="secondary",
            elem_classes=["secondary-btn"]
        )

        # ====================================================
        # FOOTER
        # ====================================================

        gr.HTML(
            """
            <div class="footer">
                DocFinder AI • Private Document
                Question Answering
            </div>
            """
        )

    # ========================================================
    # EVENTS
    # ========================================================

    process_button.click(
        fn=process_with_loader,
        inputs=[file_input],
        outputs=[
            process_loader,
            process_status
        ]
    )

    ask_button.click(
        fn=ask_with_loader,
        inputs=[question],
        outputs=[
            search_loader,
            answer_output,
            sources_output,
            search_status
        ]
    )

    question.submit(
        fn=ask_with_loader,
        inputs=[question],
        outputs=[
            search_loader,
            answer_output,
            sources_output,
            search_status
        ]
    )

    reset_button.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            file_input,
            process_status,
            search_status,
            answer_output,
            sources_output
        ]
    )


# ============================================================
# LAUNCH  (Render-compatible)
# ============================================================

demo.queue()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
