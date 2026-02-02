from pathlib import Path

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
VECTOR_DIR = KB_DIR / "faiss_index"

_vectorstore = None
_qa_chain = None


# ---------- Embeddings ----------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------- PDF Loading ----------
def load_documents():
    if not KB_DIR.exists():
        return []

    loader = DirectoryLoader(
        str(KB_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    docs = loader.load()
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


# ---------- Vector Store ----------
def rebuild_knowledge_base():
    global _vectorstore

    documents = load_documents()
    if not documents:
        return None

    embeddings = get_embeddings()
    _vectorstore = FAISS.from_documents(documents, embeddings)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore.save_local(str(VECTOR_DIR))

    return _vectorstore


def get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    if VECTOR_DIR.exists():
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            str(VECTOR_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return _vectorstore

    return None


# ---------- LLM ----------
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)


# ---------- QA Chain ----------
def get_qa_chain():
    global _qa_chain

    if _qa_chain is not None:
        return _qa_chain

    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None

    llm = get_llm()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a palmistry expert.\n"
            "Use the following context to answer accurately.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

    return _qa_chain
# Preload vectorstore at startup (Railway safe)
def preload():
    try:
        get_vectorstore()
        get_qa_chain()
        print("Vectorstore + QA chain loaded")
    except Exception as e:
        print("Preload failed:", e)
