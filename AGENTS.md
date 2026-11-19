# Instruções para agentes

## Projeto

Treinar um modelo de linguagem do zero com a obra de Machado de Assis, para
aprender como um LLM funciona. Escopo, etapas e critérios de pronto estão em
[docs/PRD.md](docs/PRD.md). RAG e agentes estão fora do escopo.

## Como trabalhar

- Responda e documente em português do Brasil. Código, identificadores e mensagens
  de commit em inglês.
- O objetivo é didático: em cada etapa, explique o conceito antes do código, escreva
  em passos pequenos e mostre o resultado antes de avançar.
- Não pule etapas nem adicione bibliotecas que escondam o mecanismo (por exemplo,
  Hugging Face Trainer ou modelos pré-treinados) sem decisão registrada.
- Registre decisões técnicas e ações do proprietário em
  [docs/DECISOES.md](docs/DECISOES.md).

## Regras obrigatórias

- Siga [GIT_POLICY.md](GIT_POLICY.md) antes de qualquer commit.
- Siga [SECURITY.md](SECURITY.md) para segredos, dados e artefatos.
- O repositório é público. Nunca inclua credenciais, dados pessoais, caminhos
  absolutos da máquina ou textos com direito autoral.
- Corpus, checkpoints e pesos nunca entram no Git.
- Não evite vazamento só "na prática": a divisão por obra e o tokenizador treinado
  só no treino são regras, e devem ter testes.
- Commit, push, mudança de visibilidade e publicação de dados ou pesos só com pedido
  explícito do proprietário.
