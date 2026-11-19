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

## 2026-11-19: Fonte principal do corpus

- **Decisão:** usar como fonte principal os `.txt` do repositório
  [luxedo/dataset-machado-de-assis](https://github.com/luxedo/dataset-machado-de-assis)
  (124 arquivos, cerca de 11,3 MB, convertidos dos PDFs do portal do MEC).
- **Como:** script próprio que baixa os arquivos e registra origem, commit e
  SHA-256 de cada um. O repositório não é copiado; a licença GPL-3.0 dele cobre o
  script, e os textos estão em domínio público.
- **Fonte complementar:** Wikisource, para conferir obras e preencher lacunas.
- **Responsável:** proprietário.

## 2026-11-19: Traduções fora do corpus principal

- **Decisão:** traduções feitas por Machado (por exemplo, *Oliver Twist* e *Os
  Trabalhadores do Mar*) ficam fora do corpus principal.
- **Motivo:** o estilo é do autor original, não de Machado.
- **Responsável:** proprietário.

## 2026-11-19: Portal do MEC sem coleta automatizada

- **Fato:** o portal `machado.mec.gov.br` responde com desafio do Cloudflare a
  clientes automatizados, e o `robots.txt` declara `Content-Signal: ai-train=no`.
- **Decisão:** não contornar a proteção nem coletar o portal por script. Os textos
  são de domínio público, mas a sinalização é a condição de acesso do operador.
- **Pendente:** comparar o inventário do luxedo com o índice do portal (consultado
  no navegador) e com o Wikisource. Se faltar algo relevante, buscar primeiro no
  Wikisource; download manual do MEC só por decisão registrada do proprietário.
- **Responsável:** proprietário.

## 2026-11-19: Extração própria a partir dos PDFs

- **Decisão:** a fonte primária passa a ser a pasta `pdf/` do repositório luxedo
  (116 PDFs do portal do MEC), extraída por script próprio com PyMuPDF. Os `.txt`
  do luxedo ficam como linha de base para medir a qualidade da nossa extração.
- **Motivo:** o script do luxedo (`textract` + regex) colapsa quebras de linha
  duplas (perde os parágrafos), remove só as marcações das notas de rodapé (o texto
  das notas, que é editorial, pode ter ficado) e junta palavras hifenizadas na quebra
  de linha ("ave-marias" virou "avemarias").
- **Regras da extração:**
  1. Descartar cabeçalhos, rodapés e números de página pela posição e repetição.
  2. Separar notas de rodapé (fonte menor, pé da página) num relatório; não entram
     no corpus.
  3. Reconstruir parágrafos pelo recuo ou espaçamento entre blocos.
  4. Hífen de fim de linha decidido pelo próprio corpus: mantém se a forma com hífen
     ocorre no meio de linha; junta se a forma sem hífen ocorre; senão, vai para uma
     lista de revisão.
  5. Cabeçalho da edição ("Texto-fonte", editora, ano) sai do texto e vira metadado.
  6. Normalizar Unicode (NFC) e ligaduras; manter o travessão de diálogo.
  7. OCR (Tesseract, modelo `por`) só para páginas sem camada de texto.
- **Verificação:** testes automáticos para cada regra e comparação por amostras com
  o PDF.
- **Responsável:** proprietário.

## 2026-11-19: Download manual do MEC usado só como conferência

- **Ação do proprietário:** baixou manualmente, pelo navegador, 58 PDFs do portal do
  MEC.
- **Resultado da comparação com a pasta `pdf/` do luxedo:** os 58 existem no luxedo
  com o mesmo nome. 15 são idênticos byte a byte; 43 diferem em 5 bytes, só nos
  metadados XMP (`http://` virou `https://` nos namespaces), sem mudança no
  conteúdo. Não há nenhum arquivo novo; os contos são as mesmas 7 coletâneas.
- **Decisão:** usar as cópias do luxedo como fonte; os arquivos baixados do MEC não
  entram no pipeline nem no repositório e servem apenas como verificação de que o
  luxedo reflete o portal.
- **Faltam nos baixados** (presentes no luxedo): 45 críticas, 10 miscelâneas e 3
  traduções.
- **Responsável:** proprietário.

## 2026-11-19: Fonte fixada em commit e verificada por hash

- **Decisão:** o download usa o commit `a13ca4a` do luxedo (único commit do
  repositório, de 2019) e baixa só `pdf/` e `txt/`. O CSV com o texto completo fica
  de fora, porque duplica os `.txt`.
- **Verificação:** cada arquivo é conferido contra o hash de blob do Git informado
  pela API do GitHub para esse commit. O catálogo
  `data/catalog/luxedo_manifest.csv` registra caminho, tamanho, hash do Git e
  SHA-256; os arquivos ficam em `data/raw/luxedo/`, fora do Git.
- **Resultado:** 232 arquivos (116 PDFs e 116 TXTs), 31,0 MB, todos verificados.
- **Responsável:** proprietário.

## 2026-11-19: pdfplumber no lugar do PyMuPDF

- **Decisão:** extrair os PDFs com `pdfplumber` (sobre o `pdfminer.six`), e não com
  PyMuPDF.
- **Motivo:** o PyMuPDF é AGPL-3.0, o que prenderia a licença do código do projeto,
  ainda não escolhida. O pdfplumber (MIT) e suas dependências (`pdfminer.six` MIT,
  `pypdfium2` BSD-3/Apache-2.0) são permissivos e dão posição, fonte e tamanho de
  cada caractere, o que a extração precisa. O `pypdfium2` também renderiza páginas,
  caso o OCR venha a ser necessário.
- **Custo:** é mais lento (cerca de 6,5 min para inspecionar os 116 PDFs na CPU).
- **Responsável:** proprietário.

## 2026-11-19: Achados da inspeção dos PDFs e revisão das regras de extração

- **Achados** (`uv run python -m machado_gpt.corpus.inspect_pdfs`; detalhes em
  [docs/etapas/01-corpus.md](etapas/01-corpus.md)):
  - Nenhuma página precisa de OCR: as 30 páginas sem texto são todas a última
    página em branco de um PDF.
  - Não há cabeçalhos, rodapés nem números de página. As linhas que se repetem
    entre páginas são conteúdo: falas no teatro, numerais de seção, epígrafes e
    assinaturas.
  - Todos os 116 PDFs abrem com uma nota da edição moderna em fonte menor (7,4 a
    7,5 contra 9,9 a 10,0 do corpo), com rótulos variados ("Texto-fonte",
    "Textos-Fonte", "Edição referência"...).
  - Parágrafos são marcados por espaçamento vertical (linha a cada 12 pt,
    parágrafo a cada 24 a 26 pt), e não por recuo.
  - Há 1.256 linhas terminadas em hífen, muitas delas hífens reais de pronome
    ("admiti-/me").
  - Há notas numeradas em fonte menor no fim de algumas obras (por exemplo,
    *Americanas*). É preciso verificar se são do autor ou do editor.
- **Revisão proposta:** a regra 1 (descartar linhas repetidas) apagaria texto
  verdadeiro e sai; a regra 7 (OCR) não é necessária para este corpus; a regra 5
  passa a identificar a nota da edição pelo tamanho da fonte, e não por palavras.
- **Confirmação:** o proprietário confirmou a revisão das regras 1, 5 e 7.
- **Responsável:** proprietário.

## 2026-11-19: Notas fora do corpus, inclusive as do autor

- **Decisão:** nenhuma nota entra no corpus: nem as de editores modernos, nem as
  notas que o próprio Machado escreveu (por exemplo, as do fim de *Americanas*).
- **Motivo:** dispensa verificar a autoria nota a nota, e as notas são texto
  explicativo, de registro diferente da obra.
- **Como:** linhas em fonte menor que o corpo (nota da edição, notas numeradas) são
  separadas do texto. A nota da edição vira metadado; as demais vão para um relatório
  de conferência, fora do corpus.
- **Responsável:** proprietário.

## 2026-11-19: Regras de parágrafo e hífen medidas em Dom Casmurro

- **Hífen:** em *Dom Casmurro*, nenhuma das 57 linhas terminadas em hífen era palavra
  partida; os PDFs vêm de HTML, que não hifeniza. Casos sem evidência mantêm o hífen
  e ficam registrados no `.json` da extração para revisão.
- **Parágrafo na quebra de página:** começa um novo parágrafo se a linha anterior
  termina em pontuação **e** é curta (antes de 90% da margem direita), ou se a
  linha seguinte começa com travessão. A regra foi escolhida por medição nas
  fronteiras dentro da página (ver [docs/etapas/01-corpus.md](etapas/01-corpus.md)).
- **Resultado:** as mesmas 66.174 palavras do `.txt` do luxedo, com 57 hífens
  corrigidos, um trecho fora de ordem corrigido e 1.718 parágrafos.
- **Responsável:** proprietário.

## 2026-11-19: Regras de extração validadas em romances e contos

- **Resultado:** as regras de *Dom Casmurro*, sem ajuste, aplicadas aos 10 romances
  e 7 contos: 835.312 palavras, todas acima de 99,8% iguais ao `.txt` do luxedo.
  Nenhuma palavra presente no luxedo falta na nossa extração. As diferenças são
  hífens grudados, trechos fora de ordem e espaços depois de hífen, todos do lado
  do luxedo.
- **Decisão:** romances e contos usam a extração atual. Poesia, teatro, crônica,
  crítica e miscelânea passam por inspeção própria antes.
- **Responsável:** proprietário.

## 2026-11-19: Epígrafes de outros autores fora do corpus

- **Decisão:** epígrafes de outros autores saem do corpus, em verso e em prosa.
  Ficam no `.json` de cada obra, para conferência.
- **Motivo:** não são texto de Machado; muitas estão em francês, latim ou espanhol;
  o mote da *Gazeta de Holanda* se repete 48 vezes e ensinaria o modelo a decorá-lo.
  É coerente com a exclusão das traduções.
- **Exceções que ficam:** dedicatórias, datas, rubricas e autocitações ("DO AUTOR").
- **Responsável:** proprietário.

## 2026-11-19: Traduções e poemas de outros autores por lista manual

- **Decisão:** poemas traduzidos por Machado e poemas de outros autores incluídos
  nos livros de poesia saem do corpus por uma lista manual versionada, com a fonte
  de cada identificação. Casos duvidosos ficam marcados para revisão do
  proprietário. A lista também cobre epígrafes que as regras não detectam (2 em
  *Americanas*).
- **Motivo:** a detecção automática falha (por exemplo, "O Corvo" e "Ser ou não ser"
  não têm marca no título).
- **Responsável:** proprietário.

## 2026-11-19: Regras de extração para poesia

- **Verso:** bloco em que nenhuma linha de continuação começa com minúscula e as
  linhas não vão até a margem; mantém uma linha por verso.
- **Estrofe na quebra de página:** em livros majoritariamente em verso, a estrofe
  continua, salvo quando o poema tem estrofes regulares (80% com o mesmo tamanho) e
  a estrofe já está completa antes da quebra. Medido: a pontuação só fecha estrofe
  em 42% dos casos.
- **Removidos:** índices (de livro e de conto), seção de notas no corpo ("[n]
  texto" até o fim), marcadores "[n]" e "[iii]", linhas inteiras entre colchetes
  (inserções do editor) e epígrafes.
- **Verificação:** 62 testes; em 952.069 palavras do luxedo, 67 ausentes sem
  explicação direta, todas identificadas como defeitos do luxedo ou material
  editorial. Detalhes em [docs/etapas/01-corpus.md](etapas/01-corpus.md).
- **Responsável:** proprietário.

## 2026-11-19: Catálogo manual de exclusões

- **Decisão:** traduções, poemas de outros autores, epígrafes sem recuo e notas de
  Machado no corpo do texto, quando as regras não os detectam com segurança, saem
  por `data/catalog/exclusions.csv`. Cada item traz a evidência (nota de Machado ou
  subtítulo no PDF). Se um item não for encontrado exatamente uma vez no texto
  extraído, a extração para com erro.
- **Resultado:** 30 itens, cerca de 11.600 palavras; a poesia passa de 109.332 para
  97.740 palavras.
- **Responsável:** proprietário.

## 2026-11-19: Poemas em francês e imitações fora do corpus

- **Decisão:** *Un vieux pays* (*Falenas*) e *Refus* (*Poesias dispersas*), escritos
  por Machado em francês, e as imitações *Minha Mãe* (de Cowper) e *O Casamento do
  Diabo* (do alemão) saem do corpus.
- **Motivo:** o corpus é em português, e poucas palavras em francês só acrescentariam
  ruído. Imitação é obra derivada de outro autor, coerente com a exclusão das
  paráfrases que o próprio Machado chama de "tradução livre".
- **Responsável:** proprietário.

## 2026-11-19: Regras de extração para teatro e correções gerais

- **Falas:** linha que começa com `NOME —` abre bloco; em teatro, o nome do
  personagem em linha própria é unido à fala (e às rubricas entre eles) por quebra
  de linha, sem acrescentar caracteres.
- **Correção geral 1, espaçamento de linha:** passa a ser medido depois de linhas
  cheias. A moda geral dava 24 pt em 9 das 10 peças e unia parágrafos; nos outros
  gêneros as duas medidas coincidem (nenhuma mudança fora do teatro).
- **Correção geral 2, espaço entre palavras:** espaços gravados no PDF são
  ignorados e o espaço é deduzido pela distância entre letras, com tolerância de
  1,5 pt (o padrão de 3 pt grudava palavras como "dado como"). Só 3 peças mudaram,
  todas para melhor.
- **Lista manual:** *Hoje avental, amanhã luva* (imitada do francês) sai inteira;
  saem também a carta de Quintino Bocaiúva, a nota final de *Os Deuses de Casaca* e
  a epígrafe de Camões em *Tu, só tu, puro amor*. Prefácios de Machado sobre as
  peças ficam.
- **Responsável:** proprietário.

## 2026-11-19: Crônica, crítica e miscelânea; ajustes gerais

- **Espaço entre palavras:** tolerância de 1,8 pt, escolhida pela distribuição das
  distâncias entre letras em todos os PDFs (letras até 1,6 pt; menor espaço real,
  2,0 pt); o espaço não separável gravado no PDF é ignorado como o espaço comum.
- **Comentários do editor** entre colchetes no meio do texto saem; palavras
  restauradas pelo editor ficam, sem colchetes.
- **Citações em verso** de outros poetas, dentro de críticas e crônicas, ficam no
  corpus (cerca de 13.600 palavras, 1,9% desses gêneros). Motivo: fazem parte do
  argumento de Machado, e removê-las deixaria frases incompletas. Registrado como
  limite conhecido.
- **Resultado:** 113 obras extraídas, 1.704.663 palavras.
- **Responsável:** proprietário.

## 2026-11-19: Divisão por obra, duplicatas e catálogo final

- **Grupos:** obras com 50 ou mais sequências de 8 palavras em comum ficam no mesmo
  conjunto (versões da mesma obra e poemas republicados).
- **Duplicatas:** obra com 90% ou mais do texto contido em outra sai do corpus
  (*Crítica Teatral* e *A Mãe*, contidas em *Revista Dramática*).
- **Divisão:** por gênero, em ordem de SHA-256 de `machado-gpt-split-v1:grupo`;
  validação e teste com ~10% das palavras de cada gênero (até 125% do alvo); se um
  conjunto ficar vazio num gênero, recebe o primeiro grupo com até 25% do gênero.
- **Resultado:** 110 obras; treino 1.300.828 palavras (77%), validação 213.246
  (13%), teste 173.198 (10%); todo gênero nos três conjuntos. Sequências de 8
  palavras da validação e do teste presentes no treino: 0,014% e 0,013%.
- **Catálogo:** `data/catalog/works.csv`, versionado; textos em `data/processed/`,
  fora do Git.
- **Responsável:** proprietário.

## 2026-11-19: Etapa 2, ambiente e convenções de medida

- **PyTorch CPU e NumPy:** dependências do projeto (Apache-2.0/BSD). Localmente usa
  a versão só CPU do PyTorch (índice `pytorch-cpu` no `pyproject.toml`), porque as
  etapas 2 a 4 rodam em CPU; o Colab tem a sua própria versão com GPU.
- **Vocabulário de caracteres só do treino,** com um símbolo de desconhecido
  (U+FFFD) para caracteres ausentes do treino (1 ocorrência na validação, `½`).
- **Medidas:** perda em nats por caractere (principal), bits por caractere, bits
  por byte UTF-8 (para comparar com o BPE) e perplexidade.
- **Resultados versionados** em `results/` (métricas e amostras, sem pesos);
  tabelas e pesos em `runs/`, fora do Git.
- **Linha de base:** bigrama por contagem, 2,383 nats/caractere (3,33 bits/byte) na
  validação. Detalhes em [docs/etapas/02-bigrama.md](etapas/02-bigrama.md).
- **Responsável:** proprietário.
