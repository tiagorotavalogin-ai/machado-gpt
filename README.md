# machado-gpt

Um modelo de linguagem (language model) pequeno, treinado do zero com a obra de Machado de Assis,
construído passo a passo para aprender como um LLM funciona por dentro.

> **Estado:** etapas 1 a 5 concluídas: corpus, bigrama, GPT por caracteres,
> tokenizador (tokenizer) BPE e treino no Colab ([notebook](notebooks/05-colab.ipynb)).
> Validação (validation), em bits por byte: bigrama 3,33, GPT por caracteres 2,22 e GPT
> BPE de 12 M de parâmetros (parameters) 1,66. A etapa 6 (pré-treino geral) é opcional.
> Diários:
> [corpus](docs/etapas/01-corpus.md), [bigrama](docs/etapas/02-bigrama.md),
> [GPT](docs/etapas/03-gpt-caracteres.md), [BPE](docs/etapas/04-tokenizador-bpe.md),
> [treino no Colab](docs/etapas/05-treino-colab.md).

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
6. (Opcional) Pré-treino em português geral e fine-tuning com Machado.

## Documentação

- [docs/visual/machado-gpt.html](docs/visual/machado-gpt.html): as etapas explicadas com
  demonstrações interativas (abra no navegador; gerada por `uv run python -m machado_gpt.visual`).
- [docs/PRD.md](docs/PRD.md): objetivo, escopo, etapas e critérios de pronto.
- [docs/REFERENCIAS.md](docs/REFERENCIAS.md): projetos e materiais de referência.
- [docs/DECISOES.md](docs/DECISOES.md): registro de decisões.
- [GIT_POLICY.md](GIT_POLICY.md): o que entra e o que não entra no repositório.
- [SECURITY.md](SECURITY.md): segredos, dados e reporte de vulnerabilidades.

## Como rodar

_A definir na etapa 1._

## Licença

_A definir pelo proprietário antes da publicação._ Os textos de Machado de Assis
estão em domínio público; a licença do código será escolhida à parte.
