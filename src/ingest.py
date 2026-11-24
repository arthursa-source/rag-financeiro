"""Etapa 1 do RAG: carregar documentos, dividir em chunks e indexar no pgvector.

Uso: python src/ingest.py
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import get_vectorstore

PASTA_DOCS = Path(__file__).resolve().parent.parent / "docs"


def carregar_documentos(pasta: Path = PASTA_DOCS) -> list[Document]:
    """Lê arquivos .md, .txt e .pdf. Guarda o nome do arquivo em metadata para citar a fonte."""
    docs = []
    for arquivo in sorted(pasta.iterdir()):
        if arquivo.suffix in (".md", ".txt"):
            texto = arquivo.read_text(encoding="utf-8")
        elif arquivo.suffix == ".pdf":
            texto = "\n".join(p.extract_text() or "" for p in PdfReader(arquivo).pages)
        else:
            continue
        docs.append(Document(page_content=texto, metadata={"fonte": arquivo.name}))
    return docs


def dividir_em_chunks(docs: list[Document]) -> list[Document]:
    """Chunking: textos grandes viram pedaços menores.

    - chunk_size=500: pedaço pequeno o bastante para ser específico na busca.
    - chunk_overlap=80: repete um trecho entre chunks vizinhos para não cortar uma ideia ao meio.
    - separators: tenta quebrar primeiro por seção (##), depois parágrafo, depois frase.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def main() -> None:
    docs = carregar_documentos()
    chunks = dividir_em_chunks(docs)
    # recriar=True apaga a coleção antiga, assim rodar duas vezes não duplica chunks.
    store = get_vectorstore(recriar=True)
    store.add_documents(chunks)
    print(f"{len(docs)} documentos -> {len(chunks)} chunks indexados no pgvector.")


if __name__ == "__main__":
    main()
