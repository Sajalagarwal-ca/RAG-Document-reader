import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

VECTOR_DB_PATH = "faiss_index"
INDEX_META_PATH = os.path.join(VECTOR_DB_PATH, "index_meta.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt"}

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)


def _iter_supported_files(folder_path: str) -> List[Path]:
    folder = Path(folder_path).expanduser().resolve()
    files: List[Path] = []
    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)
    return sorted(files)


def _load_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(file_path)).load()
    elif suffix == ".docx":
        docs = Docx2txtLoader(str(file_path)).load()
    elif suffix in {".xlsx", ".xls"}:
        docs = UnstructuredExcelLoader(str(file_path), mode="elements").load()
    elif suffix == ".csv":
        docs = CSVLoader(file_path=str(file_path), autodetect_encoding=True).load()
    else:
        docs = TextLoader(file_path=str(file_path), autodetect_encoding=True).load()

    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["source_file"] = file_path.name
        doc.metadata["source_extension"] = suffix
    return docs


def load_documents_from_folder(folder_path: str) -> Dict[str, Any]:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    all_docs: List[Document] = []
    processed_files = 0
    failed_files: List[str] = []

    for file_path in _iter_supported_files(str(folder)):
        try:
            loaded = _load_file(file_path)
            all_docs.extend(loaded)
            processed_files += 1
        except Exception as exc:  # pragma: no cover - best effort loading
            failed_files.append(f"{file_path.name}: {exc}")

    if not all_docs:
        raise ValueError(
            "No documents could be loaded from the selected folder. "
            "Add PDF/DOCX/XLSX/CSV/TXT files and try again."
        )

    return {
        "documents": all_docs,
        "files_processed": processed_files,
        "failed_files": failed_files,
    }


def _build_source_references(source_docs: List[Document]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    seen = set()
    for doc in source_docs:
        source_file = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id")
        key = (source_file, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)

        snippet = " ".join(doc.page_content.split())
        references.append(
            {
                "source_file": source_file,
                "page": page,
                "chunk_id": chunk_id,
                "snippet": snippet[:240] + ("..." if len(snippet) > 240 else ""),
            }
        )
    return references


def create_vector_db(folder_path: str) -> Dict[str, Any]:
    ingestion = load_documents_from_folder(folder_path)
    documents: List[Document] = ingestion["documents"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectordb.save_local(VECTOR_DB_PATH)

    metadata = {
        "folder_path": str(Path(folder_path).expanduser().resolve()),
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_processed": ingestion["files_processed"],
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "failed_files": ingestion["failed_files"],
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def get_index_status() -> Dict[str, Any]:
    if not os.path.exists(INDEX_META_PATH):
        return {
            "index_exists": False,
            "folder_path": None,
            "indexed_at_utc": None,
            "files_processed": 0,
            "documents_loaded": 0,
            "chunks_created": 0,
            "failed_files": [],
        }
    with open(INDEX_META_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["index_exists"] = True
    return data


def get_qa_chain(k: int = 4) -> RetrievalQA:
    if not os.path.exists(VECTOR_DB_PATH):
        raise ValueError(
            "Vector database not found. Build/refresh the knowledge base first."
        )

    vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

    prompt_template = """You are a personal assistant that answers only from the provided context.
If the context does not contain the answer, reply with exactly: "I don't know."
Keep the answer concise and useful.

CONTEXT:
{context}

QUESTION:
{question}
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=_get_llm(),
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask_question(question: str) -> Dict[str, Any]:
    chain = get_qa_chain()
    response = chain.invoke({"query": question})
    sources = _build_source_references(response.get("source_documents", []))
    return {"answer": response["result"], "sources": sources}