import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def scan_pdfs(directory: str) -> list:
    """Scan directory for PDFs case-insensitively (catches .pdf and .PDF)."""
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])


def run_ingestion(pdf_paths: list):
    """Ingest one or more PDF files into a FAISS vector index."""
    embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    db_path = "faiss_index"

    all_documents = []
    for file_path in pdf_paths:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found. Skipping.")
            continue
        print(f"Loading {file_path}...")
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())

    if not all_documents:
        print("No documents loaded. Aborting.")
        return

    # Semantic Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(all_documents)

    # Vectorization
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create and Save Index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(db_path)
    print(f"Vector database saved to '{db_path}' ({len(chunks)} chunks total).")


if __name__ == "__main__":
    # Always resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"\nScanning folder: {script_dir}")

    while True:
        # Re-scan on every loop iteration — picks up new PDFs instantly
        pdf_files = scan_pdfs(script_dir)

        if not pdf_files:
            print("\nNo PDF files found in this folder.")
            retry = input("Add a PDF then press Enter to refresh, or type 'exit' to quit: ").strip().lower()
            if retry == 'exit':
                break
            continue

        # Show numbered menu
        print("\nAvailable PDFs:")
        for i, name in enumerate(pdf_files, start=1):
            print(f"  [{i}] {name}")
        print("  [R] Refresh list")
        print("  [Q] Quit")

        choice = input(f"\nSelect a PDF to ingest (1-{len(pdf_files)}, R to refresh, Q to quit): ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'r':
            print("\nRefreshing list...")
            continue
        elif choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
            selected = pdf_files[int(choice) - 1]
            print(f"\nSelected: {selected}")
            run_ingestion([selected])
            break
        else:
            print(f"Invalid input. Enter a number 1-{len(pdf_files)}, R, or Q.")
