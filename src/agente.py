"""Agente de IA com tool/function calling (LangChain).

O LLM decide sozinho qual ferramenta chamar:
- buscar_documentos: usa o RAG para responder sobre regras do banco.
- calcular_parcela: faz a conta da Tabela Price (LLM erra conta; ferramenta não erra).

Uso: python src/agente.py "Quero 10 mil em 24 meses com score 750. Quanto fica a parcela?"
"""
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import get_llm
from rag import busca_hibrida, formatar_contexto


@tool
def buscar_documentos(pergunta: str) -> str:
    """Busca nas políticas e FAQs do banco (crédito, Pix, tarifas). Use para qualquer regra, taxa ou limite."""
    return formatar_contexto(busca_hibrida(pergunta))


@tool
def calcular_parcela(valor: float, taxa_mensal_percentual: float, meses: int) -> str:
    """Calcula a parcela fixa de um empréstimo pela Tabela Price.

    Args:
        valor: valor emprestado em reais.
        taxa_mensal_percentual: juros ao mês em %, ex.: 1.9 para 1,9%.
        meses: número de parcelas.
    """
    return f"Parcela mensal: R$ {parcela_price(valor, taxa_mensal_percentual, meses):.2f}"


def parcela_price(valor: float, taxa_mensal_percentual: float, meses: int) -> float:
    """Fórmula Price: PMT = PV * i / (1 - (1 + i)^-n)."""
    i = taxa_mensal_percentual / 100
    if i == 0:
        return valor / meses
    return valor * i / (1 - (1 + i) ** -meses)


SYSTEM_PROMPT = (
    "Você é um assistente de crédito de um banco. "
    "Sempre consulte buscar_documentos antes de afirmar regras, taxas ou limites. "
    "Nunca peça ao cliente uma informação que pode estar nos documentos: busque primeiro "
    "(ex.: a taxa de juros depende do score e está na política de crédito). "
    "Para qualquer conta de parcela, use calcular_parcela com a taxa encontrada. "
    "Responda em português, de forma curta, e cite a fonte do documento."
)


def criar_agente():
    return create_agent(get_llm(), tools=[buscar_documentos, calcular_parcela], system_prompt=SYSTEM_PROMPT)


def perguntar(pergunta: str) -> str:
    resultado = criar_agente().invoke({"messages": [{"role": "user", "content": pergunta}]})
    return resultado["messages"][-1].text


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or "Quero 10 mil em 24 meses e meu score é 750. Quanto fica a parcela?"
    print(perguntar(pergunta))
