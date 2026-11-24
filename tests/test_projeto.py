"""Testes do projeto.

- Testes unitários (sem API): rodam sempre.
- Avaliação do RAG (com API + banco): mede se a resposta traz o fato certo e a fonte certa.
  Rode com: pytest -m avaliacao
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agente import parcela_price  # noqa: E402
from ingest import carregar_documentos, dividir_em_chunks  # noqa: E402


# ---------- Testes unitários ----------

def test_parcela_price_valor_conhecido():
    # 10.000 a 1,9% a.m. em 24 meses: parcela ~ R$ 522,74 (Tabela Price).
    assert parcela_price(10_000, 1.9, 24) == pytest.approx(522.74, abs=0.01)


def test_parcela_sem_juros():
    assert parcela_price(1200, 0, 12) == 100


def test_chunks_tem_fonte_e_tamanho_limitado():
    chunks = dividir_em_chunks(carregar_documentos())
    assert len(chunks) > 3
    for c in chunks:
        assert c.metadata["fonte"].endswith(".md")
        assert len(c.page_content) <= 500


# ---------- Avaliação de qualidade do RAG ----------
# Cada caso: pergunta, trecho que PRECISA aparecer na resposta, arquivo que deve ser a fonte.
CASOS = [
    ("Qual o limite do Pix entre 20h e 6h?", "1.000", "faq_pix.md"),
    ("Qual o prazo máximo do crédito pessoal?", "48", "politica_credito.md"),
    ("Quanto custa um saque extra?", "2,50", "tarifas_conta.md"),
    ("Em quantos dias posso contestar um golpe no Pix?", "80", "faq_pix.md"),
]


@pytest.mark.avaliacao
@pytest.mark.parametrize("pergunta,esperado,fonte", CASOS)
def test_rag_responde_com_fato_e_fonte(pergunta, esperado, fonte):
    from rag import responder

    resultado = responder(pergunta)
    assert esperado in resultado["resposta"]
    assert fonte in resultado["fontes"]


@pytest.mark.avaliacao
def test_rag_nao_inventa_fora_do_contexto():
    from rag import responder

    resultado = responder("Qual a taxa do financiamento imobiliário?")
    assert "Não encontrei" in resultado["resposta"]
