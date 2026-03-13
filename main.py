import os
from datetime import datetime

import streamlit as st

from langchain_helper import ask_question, create_vector_db, get_index_status

st.set_page_config(page_title="Personal Assistant RAG", page_icon=":bookmark_tabs:")
st.title("Personal Assistant RAG")
st.caption("Ask questions over your personal folder (PDF/DOCX/XLSX/CSV/TXT).")

default_folder = os.path.join(os.getcwd(), "Directory")
if "folder_path" not in st.session_state:
    st.session_state["folder_path"] = default_folder

folder_path = st.text_input(
    "Personal documents folder path",
    value=st.session_state["folder_path"],
    help="This folder is scanned recursively for supported files.",
)
st.session_state["folder_path"] = folder_path

if st.button("Build / Refresh Knowledge Base"):
    with st.spinner("Indexing documents..."):
        try:
            metadata = create_vector_db(folder_path)
            st.success("Knowledge base updated.")
            if metadata.get("failed_files"):
                st.warning("Some files could not be indexed.")
                for failed in metadata["failed_files"]:
                    st.write(f"- {failed}")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

status = get_index_status()
st.subheader("Index Status")
if status.get("index_exists"):
    indexed_at = status.get("indexed_at_utc")
    pretty_time = indexed_at
    if indexed_at:
        try:
            pretty_time = datetime.fromisoformat(
                indexed_at.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            pretty_time = indexed_at

    st.write(f"**Folder:** `{status.get('folder_path')}`")
    st.write(f"**Last refresh:** {pretty_time}")
    st.write(f"**Files processed:** {status.get('files_processed', 0)}")
    st.write(f"**Chunks created:** {status.get('chunks_created', 0)}")
else:
    st.info("No knowledge base found yet. Build it from your personal folder first.")

question = st.text_input("Ask your assistant")
if question:
    with st.spinner("Thinking..."):
        try:
            result = ask_question(question)
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("References")
            references = result.get("sources", [])
            if references:
                for idx, ref in enumerate(references, start=1):
                    source_file = ref.get("source_file", "Unknown")
                    page = ref.get("page")
                    chunk_id = ref.get("chunk_id")
                    snippet = ref.get("snippet", "")
                    location_bits = [f"file: `{source_file}`"]
                    if page is not None:
                        location_bits.append(f"page: `{page}`")
                    if chunk_id is not None:
                        location_bits.append(f"chunk: `{chunk_id}`")
                    st.markdown(f"{idx}. " + " | ".join(location_bits))
                    if snippet:
                        st.caption(snippet)
            else:
                st.caption("No references were returned.")
        except Exception as exc:
            st.error(f"Query failed: {exc}")






