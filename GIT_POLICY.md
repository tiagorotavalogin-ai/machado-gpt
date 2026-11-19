# Política de Git

Este repositório é público: o que entrar em um commit pode ser lido por qualquer
pessoa e não pode ser recolhido depois. Na dúvida, não commite e pergunte.

## O que não entra

- **Credenciais:** senhas, tokens (GitHub, Hugging Face, Colab, Weights & Biases),
  cookies, chaves e arquivos `.env`.
- **Dados pessoais:** nomes, e-mails, telefones e documentos de qualquer pessoa.
- **Dados da máquina:** caminhos absolutos, nomes de usuário locais, nomes de
  servidores privados e IDs de pastas pessoais do Google Drive.
- **Corpus e artefatos de treino:** `data/raw/`, `data/interim/`,
  `data/processed/`, `checkpoints/`, pesos (`*.pt`, `*.safetensors`, `*.gguf`...).
  Já estão no `.gitignore`.
- **Textos com direito autoral:** notas, prefácios, introduções e anotações de
  edições modernas, mesmo junto de textos em domínio público.
- **Saídas de notebooks com dados privados:** limpe as saídas antes de commitar.

Essas regras valem também para exemplos, figuras, capturas de tela e logs.

## O que entra

- Código, testes e notebooks sem saídas sensíveis.
- O catálogo de fontes do corpus (origem, edição, licença e hash), sem o texto.
- Documentação, figuras e métricas de treino.

## Mensagens de commit

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
`<tipo>(<escopo>): <descrição>`, com o escopo opcional.

- **Tipos:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `build`,
  `chore` e `revert`.
- **Escopos:** `corpus`, `tokenizer`, `model`, `train`, `eval` e `docs`.
- **Descrição:** em inglês, no imperativo, em minúsculas, sem ponto final. A linha
  inteira tem no máximo 72 caracteres.
- **Um commit, uma intenção:** separe mudanças sem relação em commits diferentes.

Exemplo: `feat(corpus): split works by deterministic hash`

## Antes de cada commit

1. Adicione os arquivos pelo nome, e não com `git add .`.
2. Revise a lista com `git status` e o conteúdo com `git diff --staged`.
3. Rode os testes, se o código mudou (comando definido na etapa 1).
4. Procure e-mails, caminhos da máquina e tokens no que vai entrar. O comando lista
   só os nomes dos arquivos, para não imprimir o que foi encontrado:

   ```bash
   git grep --cached -l -I -i -E "[a-z0-9._-]+@[a-z0-9-]+\.[a-z]{2,}|/home/|/Users/|/mnt/c/|wsl\.localhost|C:\\\\Users|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}" -- . ":!GIT_POLICY.md"
   ```

5. Confira a identidade com `git config user.name` e `git config user.email`. Use o
   e-mail `noreply` da conta do GitHub, porque o nome e o e-mail ficam públicos em
   cada commit.

Uma busca por padrões é indício, e não prova: ela não encontra nomes de pessoas nem
todos os formatos de segredo. Ao relatar o resultado, diga o que foi verificado e o
que ficou de fora.

## Se algo sensível entrar em um commit

Apagar o arquivo em um commit novo não basta, porque ele continua no histórico.
Prepare uma reescrita do histórico ou um repositório limpo e troque qualquer
credencial exposta. Se o conteúdo já foi publicado, trate-o como público.

## Antes de tornar o repositório público

- Revise todo o histórico, os metadados dos commits, os arquivos rastreados e os
  ignorados.
- Escolha uma licença: um repositório público não é, por isso, código aberto.
- Ative o canal privado de reporte descrito em [SECURITY.md](SECURITY.md).
- Ative secret scanning, push protection, Dependabot e proteção da branch `main`.
