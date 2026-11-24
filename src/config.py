"""Configuração central: chaves, modelo de LLM, embeddings e banco vetorial."""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

API_KEY = os.environ["GEMINI_API_KEY"]
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5433/rag")
COLECAO = "documentos_financeiros"

MODELO_LLM = "gemini-2.5-flash"
MODELO_EMBEDDING = "models/gemini-embedding-001"


def get_llm(temperatura: float = 0.0) -> ChatGoogleGenerativeAI:
    # Temperatura 0: respostas mais estáveis, importante para testes repetíveis.
    return ChatGoogleGenerativeAI(model=MODELO_LLM, google_api_key=API_KEY, temperature=temperatura)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING, google_api_key=API_KEY)


def get_vectorstore(recriar: bool = False) -> PGVector:
    # PGVector guarda cada chunk + seu vetor numa tabela do Postgres (extensão pgvector).
    return PGVector(
        embeddings=get_embeddings(),
        connection=DATABASE_URL,
        collection_name=COLECAO,
        pre_delete_collection=recriar,
        use_jsonb=True,
    )
