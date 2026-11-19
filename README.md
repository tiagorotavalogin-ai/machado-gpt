# machado-gpt

Um modelo de linguagem pequeno, treinado do zero com a obra de Machado de Assis,
construído passo a passo para aprender como um LLM funciona por dentro.

> **Estado:** etapa 1 (corpus). Extração completa: 113 obras, 1,7 milhão de
> palavras, com traduções e textos de outros autores removidos. Próximo passo:
> catálogo das obras, divisão treino/validação/teste por obra e estatísticas. O
> passo a passo explicado está em [docs/etapas/01-corpus.md](docs/etapas/01-corpus.md).

## O que é

Um Transformer decoder causal escrito em PyTorch puro, sem pesos pré-treinados e
sem bibliotecas de treino prontas. O caminho segue a aula
["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY),
de Andrej Karpathy, trocando Shakespeare por Machado.

O modelo vai aprender o **estilo** do autor. Ele **não** conversa, não responde
perguntas e não sabe fatos: é um gerador de texto educacional. Veja os limites em
[docs/PRD.md](docs/PRD.md).

## Etapas

1. Corpus: coleta, limpeza, procedência e divisão por obra.
2. Modelo bigrama.
3. GPT por caracteres.
4. Tokenizador BPE próprio.
5. Treino em GPU e avaliação.
6. (Opcional) Pré-treino em português geral e fine-tuning com Machado.

## Documentação

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
