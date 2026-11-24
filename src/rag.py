"""Etapa 2 do RAG: buscar os chunks relevantes (retrieval) e gerar a resposta com o LLM.

Uso: python src/rag.py "Qual o limite do Pix à noite?"
"""
import sys

import psycopg
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import COLECAO, DATABASE_URL, get_llm, get_vectorstore

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente de atendimento de um banco. "
     "Responda SOMENTE com base no contexto abaixo. "
     "Se a resposta não estiver no contexto, diga: 'Não encontrei essa informação nos documentos.' "
     "Seja direto e cite a fonte entre colchetes no final, por exemplo [faq_pix.md].\n\n"
     "Contexto:\n{contexto}"),
    ("human", "{pergunta}"),
])


def busca_vetorial(pergunta: str, k: int = 4) -> list[Document]:
    """Busca semântica: compara o embedding da pergunta com o embedding de cada chunk."""
    return get_vectorstore().similarity_search(pergunta, k=k)


def busca_palavra_chave(pergunta: str, k: int = 4) -> list[Document]:
    """Busca por palavra-chave com o full-text search do próprio Postgres.

    Pega termos exatos (ex.: 'MED', 'TED') que a busca semântica às vezes perde.
    """
    sql = """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = %(colecao)s
          AND to_tsvector('portuguese', e.document) @@ websearch_to_tsquery('portuguese', %(q)s)
        ORDER BY ts_rank(to_tsvector('portuguese', e.document),
                         websearch_to_tsquery('portuguese', %(q)s)) DESC
        LIMIT %(k)s
    """
    # websearch_to_tsquery usa AND entre termos; trocamos por OR para ser mais tolerante.
    termos = " or ".join(p for p in pergunta.replace("?", "").split() if len(p) > 2)
    url = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url) as conn:
        linhas = conn.execute(sql, {"colecao": COLECAO, "q": termos, "k": k}).fetchall()
    return [Document(page_content=texto, metadata=meta) for texto, meta in linhas]


def busca_hibrida(pergunta: str, k: int = 4) -> list[Document]:
    """Junta as duas buscas com Reciprocal Rank Fusion (RRF).

    Cada documento ganha 1/(60 + posição) em cada lista onde aparece.
    Quem aparece bem nas duas listas sobe para o topo.
    """
    pontos: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for lista in (busca_vetorial(pergunta, k), busca_palavra_chave(pergunta, k)):
        for posicao, doc in enumerate(lista):
            chave = doc.page_content
            pontos[chave] = pontos.get(chave, 0) + 1 / (60 + posicao)
            docs[chave] = doc
    ordem = sorted(pontos, key=pontos.get, reverse=True)
    return [docs[c] for c in ordem[:k]]


def formatar_contexto(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('fonte')}]\n{d.page_content}" for d in docs)


def responder(pergunta: str) -> dict:
    """Pipeline completo: retrieval híbrido + geração. Retorna resposta e fontes usadas."""
    docs = busca_hibrida(pergunta)
    cadeia = PROMPT | get_llm()
    resposta = cadeia.invoke({"contexto": formatar_contexto(docs), "pergunta": pergunta})
    return {
        "resposta": resposta.text,
        "fontes": sorted({d.metadata.get("fonte") for d in docs}),
    }


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or "Qual o limite do Pix à noite?"
    resultado = responder(pergunta)
    print(resultado["resposta"])
    print("\nFontes consultadas:", ", ".join(resultado["fontes"]))
