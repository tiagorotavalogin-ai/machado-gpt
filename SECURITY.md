# Política de segurança: preliminar

Este projeto ainda não tem versão publicada, modelo publicado nem serviço em
produção.

## Reporte privado

Antes de convidar uso externo, o proprietário deve habilitar o **GitHub Private
Vulnerability Reporting** na aba de segurança do repositório e testar o recebimento
de um reporte.

**Estado: canal ainda não configurado nem testado.** Não use este documento como
evidência de que já existe um canal ativo.

Não abra issue pública com credenciais ou detalhes que permitam explorar uma
vulnerabilidade.

## Segredos

- Segredos locais ficam no `.env`, ignorado pelo Git. O `.env.example` lista só
  nomes, nunca valores.
- No Colab, use o gerenciador de segredos do próprio Colab. Nunca cole tokens em
  células de notebook que serão commitadas.
- Use tokens separados, revogáveis e com o menor escopo possível. Para o Hugging
  Face, token de leitura no treino e token de escrita só no momento de publicar.
- No GitHub Actions, se houver: permissões mínimas declaradas, actions de terceiros
  fixadas por SHA, nenhum segredo exposto a pull requests de forks.

## Logs e artefatos

- Nunca registre tokens, cabeçalhos de autorização ou caminhos pessoais em logs.
- Checkpoints e pesos são carregados apenas a partir de fontes próprias e
  verificados por SHA-256. Prefira `safetensors`; arquivos `pickle` (`.pt`) de
  terceiros podem executar código ao serem abertos.

## Dados

- O corpus contém apenas textos em domínio público, com procedência registrada.
- Notas e anotações de edições modernas são removidas antes de entrar no corpus.
- Arquivos baixados são tratados como dados não confiáveis: ficam em pasta própria e
  nunca são executados.

## Se uma credencial vazar

1. Revogue ou troque a credencial imediatamente.
2. Revise os logs de auditoria do serviço afetado.
3. Remova o segredo dos arquivos e, se necessário, do histórico do Git.
4. Registre a causa e a prevenção em [docs/DECISOES.md](docs/DECISOES.md).
5. Lembre que apagar do último commit não invalida um segredo já exposto.
