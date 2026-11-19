# Referências

## Aula base

- **Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"**
  ([vídeo](https://www.youtube.com/watch?v=kCc8FmEb1nY),
  [nanoGPT](https://github.com/karpathy/nanoGPT)). Constrói um Transformer decoder
  e treina em Shakespeare por caracteres. É o roteiro das etapas 2 e 3.

## Projetos de referência

### LLM Legislativo Brasileiro (Baruque Rodrigues)

- Código: [github.com/BaruqueRodrigues/llm-legislativo](https://github.com/BaruqueRodrigues/llm-legislativo)
- Pesos: [huggingface.co/BaruqRodrigues/llm-legislativo](https://huggingface.co/BaruqRodrigues/llm-legislativo)
- Datasets: [huggingface.co/BaruqRodrigues](https://huggingface.co/BaruqRodrigues)
- Fio de apresentação: [x.com/baruqrodrigues/status/2110158238549155954](https://x.com/baruqrodrigues/status/2110158238549155954)

Resumo do fio (outubro de 2026):

- Corpus de mais de 930 mil documentos do Congresso (mais de 360 GiB brutos),
  publicado em 5 datasets no Hugging Face.
- OCR com PDFKit (texto digital) e Apple Vision (escaneados).
- Divisão sem vazamento: versões, apensamentos e substitutivos ligados num grafo
  (igraph) formam uma família; a divisão 80/10/10 usa hash determinístico da família.
- Tokenizador byte-level BPE de 8.192 tokens, treinado só no treino, com round-trip
  exato em UTF-8.
- Decoder causal em PyTorch puro: 6 camadas, 6 cabeças, d_model 384, contexto 512,
  weight tying.
- Perda como soma da cross-entropy dividida pelo total de alvos válidos, para não
  superponderar sequências curtas com padding.
- Treino numa A100 40GB no Colab: 173.209 passos, cerca de 20 horas, 1,24 bilhão de
  tokens; loss de validação 2,2111 (perplexidade 9,13); 0,729 nats/byte no teste.
- Modelo com cerca de 72 milhões de parâmetros. O autor relatou um erro numa etapa e
  planeja retreinar com 275 a 370 milhões. Ler o código com olhar crítico.

**O que aproveitamos:** a divisão por família (aqui, por obra), o tokenizador treinado
só no treino, a perda normalizada por alvos válidos e a métrica por byte.

### GPT4R (Alexandre Galvão Patriota)

- Código: [github.com/AGPatriota/GPT4R](https://github.com/AGPatriota/GPT4R)
- Tweets: [apresentação](https://x.com/agpatriota/status/1650136316196036614),
  [arquitetura](https://x.com/agpatriota/status/1650139443578191875)

GPT implementado em R (`torch`), com o decoder em menos de 40 linhas, treinado em
Shakespeare: contexto 256, embedding 384, 6 camadas, 6 cabeças, cerca de 10,8 milhões
de parâmetros. O autor também apresenta modelos de linguagem como regressão
multinomial, uma leitura estatística útil para entender a perda.

**O que aproveitamos:** a configuração de partida e a leitura estatística do treino.

## Fontes candidatas do corpus

A verificar na etapa 1 (edição, licença e presença de notas de editores):

- Portal Machado de Assis (MEC)
- Biblioteca Digital Domínio Público
- Wikisource em português
- Projeto Gutenberg
