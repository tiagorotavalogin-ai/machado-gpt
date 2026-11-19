# Registro de decisões

Decisões técnicas e ações do proprietário (visibilidade, licença, credenciais,
publicação de dados ou pesos). Uma entrada por decisão, mais recente no fim.

## 2026-11-19: Escopo do projeto

- **Decisão:** treinar um modelo do zero com a obra de Machado de Assis, seguindo a
  aula do Karpathy e o projeto LLM Legislativo como referências.
- **Fora do escopo:** RAG e agente conversacional, que ficam para um projeto futuro.
- **Responsável:** proprietário.

## 2026-11-19: Repositório público

- **Decisão:** o repositório será público no GitHub.
- **Pendente antes de publicar:** licença do código, revisão do histórico,
  `SECURITY.md` com canal privado ativo e recursos de segurança do GitHub ativados.
- **Responsável:** proprietário.

## 2026-11-19: Dados e pesos fora do Git

- **Decisão:** corpus, checkpoints e pesos não são versionados no Git (`.gitignore`).
  O repositório guarda o código que baixa e processa o corpus e o catálogo de fontes.
- **Motivo:** tamanho e controle de procedência.
- **Responsável:** proprietário.
