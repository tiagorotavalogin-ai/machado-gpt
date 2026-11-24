# machado-gpt

Um modelo de linguagem (language model) pequeno, treinado do zero com a obra de Machado de Assis,
construído passo a passo para aprender como um LLM funciona por dentro.

> **Estado:** as seis etapas estão concluídas. Validação (validation) de Machado, em bits
> por byte: bigrama 3,33, GPT por caracteres 2,22, GPT BPE de 12 M de parâmetros
> (parameters) 1,66 e o mesmo modelo com pré-treino (pretraining) no Wikisource e ajuste
> fino (fine-tuning) com Machado 1,59.
> Diários:
> [corpus](docs/etapas/01-corpus.md), [bigrama](docs/etapas/02-bigrama.md),
> [GPT](docs/etapas/03-gpt-caracteres.md), [BPE](docs/etapas/04-tokenizador-bpe.md),
> [treino no Colab](docs/etapas/05-treino-colab.md),
> [pré-treino](docs/etapas/06-pretreino.md).

## O que é

Um Transformer decoder causal escrito em PyTorch puro, sem pesos (weights) pré-treinados e
sem bibliotecas de treino (training) prontas. O caminho segue a aula
["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY),
de Andrej Karpathy, trocando Shakespeare por Machado.

O modelo vai aprender o **estilo** do autor. Ele **não** conversa, não responde
perguntas e não sabe fatos: é um gerador de texto educacional. Veja os limites em
[docs/PRD.md](docs/PRD.md).

## Etapas

1. Corpus: coleta, limpeza, procedência (provenance) e divisão (split) por obra.
2. Modelo bigrama.
3. GPT por caracteres.
4. Tokenizador BPE próprio.
5. Treino em GPU e avaliação.
6. Pré-treino no Wikisource em português, sem Machado, e ajuste fino com Machado.

## Documentação

- [docs/visual/machado-gpt.html](docs/visual/machado-gpt.html): as etapas explicadas com
  demonstrações interativas (abra no navegador; gerada por `uv run python -m machado_gpt.visual`).
- [docs/PRD.md](docs/PRD.md): objetivo, escopo, etapas e critérios de pronto.
- [docs/REFERENCIAS.md](docs/REFERENCIAS.md): projetos e materiais de referência.
- [docs/DECISOES.md](docs/DECISOES.md): registro de decisões.
- [GIT_POLICY.md](GIT_POLICY.md): o que entra e o que não entra no repositório.
- [SECURITY.md](SECURITY.md): segredos, dados e reporte de vulnerabilidades.

## Como rodar

Requisitos: Python 3.12 e [uv](https://docs.astral.sh/uv/). As etapas 1 a 4 rodam na CPU;
as etapas 5 e 6 treinam na GPU do Google Colab, pelos notebooks. Corpus, tokens e
checkpoints são gerados localmente e não entram no Git.

```bash
uv sync                                          # ambiente e dependências
uv run pytest                                    # testes

# Etapa 1: corpus de Machado (baixa ~31 MB, conferidos por hash)
uv run python -m machado_gpt.corpus.download
uv run python -m machado_gpt.corpus.extract $(cd data/raw/luxedo/pdf && ls */*.pdf)  # todas (~6 min)
uv run python -m machado_gpt.corpus.build        # catálogo e divisão por obra

# Etapas 2 a 4: bigrama, GPT por caracteres e tokenizador BPE
uv run python -m machado_gpt.bigram
uv run python -m machado_gpt.train_char          # ~25 min na CPU
uv run python -m machado_gpt.tokenize_corpus     # treina o BPE no treino e codifica

# Etapa 6: corpus geral (dump do Wikisource, ~88 MB, sem Machado)
uv run python -m machado_gpt.build_general       # ~12 min
uv run python -m machado_gpt.tokenize_general

# Página visual com as etapas
uv run python -m machado_gpt.visual
```

Os treinos das etapas 5 e 6 estão em [notebooks/05-colab.ipynb](notebooks/05-colab.ipynb)
e [notebooks/06-colab.ipynb](notebooks/06-colab.ipynb). Cada notebook regenera os dados no
Colab e confere, por SHA-256, que são idênticos aos registrados em `results/`.

## Licença

Código e documentação sob a [licença MIT](LICENSE). Os textos de Machado de Assis estão
em domínio público; o corpus e os pesos (weights) não são distribuídos neste repositório.
