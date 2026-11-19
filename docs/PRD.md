# PRD: machado-gpt

## Objetivo

Aprender, na prática e etapa por etapa, como se treina um modelo de linguagem do
zero: dados, tokenização, arquitetura Transformer, treino e avaliação. O corpus é a
obra de Machado de Assis, em domínio público.

O produto principal é o **entendimento**. O modelo é o meio. Cada etapa só avança
quando o conceito foi explicado, o código foi escrito e o resultado foi observado.

## Público

O proprietário do projeto, como estudo. O repositório é público para servir de
material para outras pessoas que queiram seguir o mesmo caminho.

## Escopo

### Dentro

- Coleta, limpeza e catalogação do corpus com procedência e licença de cada texto.
- Divisão treino/validação/teste **por obra**, sem vazamento.
- Modelo bigrama como linha de base.
- GPT por caracteres seguindo a aula do Karpathy.
- Tokenizador byte-level BPE treinado só com a partição de treino.
- Treino em GPU (Google Colab), com checkpoints verificáveis.
- Avaliação com loss, perplexidade e bits/nats por byte, comparável entre tokenizadores.
- Opcional: pré-treino em corpus geral em português e fine-tuning com Machado.

### Fora

- RAG, busca semântica e agentes conversacionais. Ficam para um projeto futuro.
- Ajuste por instruções, chat ou alinhamento.
- Uso de pesos pré-treinados de terceiros no modelo principal.
- Bibliotecas de treino prontas (por exemplo, Hugging Face Trainer). O loop de
  treino é escrito à mão.

## Limites esperados

O corpus completo de Machado deve ficar na ordem de milhões de tokens, centenas de
vezes menor que o do projeto de referência legislativo (1,24 bilhão). Um modelo
treinado só com ele deve reproduzir vocabulário, pontuação e ritmo, mas não terá
coerência longa nem conhecimento de mundo. Esse limite é parte do aprendizado e
será medido, não escondido.

## Etapas

| # | Etapa | Pronto quando |
|---|---|---|
| 1 | **Corpus** | Catálogo com fonte, edição, licença e hash de cada texto; textos limpos sem notas ou prefácios de editores modernos; divisão por obra reproduzível (hash determinístico); estatísticas do corpus (caracteres, palavras, obras por gênero). |
| 2 | **Bigrama** | Modelo de contagem e versão em PyTorch; loss de validação registrada como linha de base; amostras geradas. |
| 3 | **GPT por caracteres** | Atenção, blocos, normalização e resíduos implementados e explicados; loss de validação abaixo do bigrama; amostras comparadas. |
| 4 | **Tokenizador BPE** | Byte-level BPE treinado só no treino; round-trip exato em UTF-8 testado; tamanho de vocabulário justificado. |
| 5 | **Treino e avaliação** | Treino no Colab com checkpoints e hash SHA-256; métricas por byte para comparar com a etapa 3; relatório de resultados com amostras. |
| 6 | **(Opcional) Pré-treino geral** | Corpus geral em português com licença verificada; comparação entre "só Machado" e "geral + Machado". |

## Princípios

- **Reprodutibilidade:** seeds fixas, divisões determinísticas, versões registradas.
- **Sem vazamento:** nenhuma obra (nem versões dela) aparece em dois conjuntos; o
  tokenizador nunca vê validação ou teste.
- **Procedência:** todo texto tem origem e licença registradas antes de entrar.
- **Medir antes de afirmar:** toda melhoria é comparada com a etapa anterior.

## Ambiente

- Etapas 1 a 4: CPU local (WSL).
- Etapa 5 em diante: GPU no Google Colab.
- Corpus, checkpoints e pesos **não** entram no Git; a publicação deles, se houver,
  será decidida e registrada em [DECISOES.md](DECISOES.md).

## Questões em aberto

- Fontes e edições do corpus (portal do MEC, Domínio Público, Wikisource, Gutenberg).
- Ortografia original ou atualizada.
- Se e onde publicar corpus processado e pesos (por exemplo, Hugging Face).
- Licença do código.
