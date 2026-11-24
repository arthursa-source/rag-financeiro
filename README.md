# rag-financeiro

Assistente que responde dúvidas de clientes de banco (crédito, Pix, tarifas) usando documentos internos como base, e diz de qual documento tirou a resposta.

Fiz esse projeto pra estudar RAG na prática. Os documentos em `docs/` são inventados, de um banco fictício.

## Como funciona

1. `src/ingest.py` lê os documentos, quebra em pedaços de até 500 caracteres e salva os embeddings no Postgres com pgvector.
2. `src/rag.py` recebe a pergunta e busca os pedaços mais relevantes de dois jeitos: por similaridade (embedding) e por palavra-chave (full-text do Postgres). Depois junta os dois resultados com Reciprocal Rank Fusion e manda pro Gemini responder.
3. `src/agente.py` é um agente LangChain com duas ferramentas: uma que faz a busca acima e outra que calcula parcela de empréstimo pela Tabela Price.

Exemplo do agente:

```
> Quero 10 mil em 24 meses e meu score é 750. Quanto fica a parcela?
A parcela fica R$ 522,74, com taxa de 1,9% ao mês para score acima de 700 [politica_credito.md].
```

Ele buscou a taxa na política de crédito e só depois chamou a calculadora.

## Rodando

Precisa de Python 3.11+, Docker e uma chave do Gemini.

```bash
cp .env.example .env    # coloca a GEMINI_API_KEY
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
python src/rag.py "Qual o limite do Pix à noite?"
python src/agente.py "Quero 10 mil em 24 meses e meu score é 750. Quanto fica a parcela?"
```

## Testes

`pytest` roda os testes que não usam API (fórmula da parcela e chunking).

`pytest -m avaliacao` chama o modelo de verdade com algumas perguntas que já têm resposta conhecida e confere se a resposta veio certa e com a fonte certa. Também tem um caso de pergunta que não está nos documentos, pra garantir que ele responde "não encontrei" em vez de inventar.

## Algumas escolhas

- Usei pgvector em vez de Chroma porque assim dá pra fazer a busca vetorial e a de palavra-chave no mesmo banco.
- A busca por palavra-chave entrou porque a vetorial às vezes não pega siglas como "MED" ou "TED".
- Conta de parcela fica numa função Python testada, não no LLM, porque modelo erra conta.
- Temperatura 0 pra resposta não mudar entre execuções e os testes ficarem estáveis.

## O que ainda quero fazer

- Ler PDF de verdade guardando o número da página
- Reranking dos resultados
- Expor como API com FastAPI
