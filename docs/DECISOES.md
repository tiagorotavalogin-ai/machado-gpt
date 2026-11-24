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

## 2026-11-19: Dados e pesos (weights) fora do Git

- **Decisão:** corpus, checkpoints e pesos não são versionados no Git (`.gitignore`).
  O repositório guarda o código que baixa e processa o corpus e o catálogo de fontes.
- **Motivo:** tamanho e controle de procedência (provenance).
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
  do luxedo ficam como linha de base (baseline) para medir a qualidade da nossa extração.
- **Motivo:** o script do luxedo (`textract` + regex) colapsa quebras de linha
  duplas (perde os parágrafos), remove só as marcações das notas de rodapé (o texto
  das notas, que é editorial, pode ter ficado) e junta palavras hifenizadas na quebra
  de linha ("ave-marias" virou "avemarias").
- **Regras da extração:**
  1. Descartar cabeçalhos (headers), rodapés (footers) e números de página (page numbers) pela posição e repetição.
  2. Separar notas de rodapé (fonte menor, pé da página) num relatório; não entram
     no corpus.
  3. Reconstruir parágrafos pelo recuo (indentation) ou espaçamento entre blocos.
  4. Hífen de fim de linha decidido pelo próprio corpus: mantém se a forma com hífen
     ocorre no meio de linha; junta se a forma sem hífen ocorre; senão, vai para uma
     lista de revisão.
  5. Cabeçalho da edição ("Texto-fonte", editora, ano) sai do texto e vira metadado.
  6. Normalizar Unicode (NFC) e ligaduras (ligatures); manter o travessão de diálogo.
  7. OCR (Tesseract, modelo `por`) só para páginas sem camada de texto (text layer).
- **Verificação:** testes automáticos para cada regra e comparação por amostras (samples) com
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
  passa a identificar a nota da edição pelo tamanho da fonte (font size), e não por palavras.
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
- **Parágrafo na quebra de página (page break):** começa um novo parágrafo se a linha anterior
  termina em pontuação **e** é curta (antes de 90% da margem direita (right margin)), ou se a
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
  2,0 pt); o espaço não separável (non-breaking space) gravado no PDF é ignorado como o espaço comum.
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
- **Duplicatas (duplicates):** obra com 90% ou mais do texto contido em outra sai do corpus
  (*Crítica Teatral* e *A Mãe*, contidas em *Revista Dramática*).
- **Divisão:** por gênero, em ordem de SHA-256 de `machado-gpt-split-v1:grupo`;
  validação (validation) e teste com ~10% das palavras de cada gênero (até 125% do alvo); se um
  conjunto ficar vazio num gênero, recebe o primeiro grupo com até 25% do gênero.
- **Resultado:** 110 obras; treino (training) 1.300.828 palavras (77%), validação 213.246
  (13%), teste 173.198 (10%); todo gênero nos três conjuntos. Sequências de 8
  palavras da validação e do teste presentes no treino: 0,014% e 0,013%.
- **Catálogo:** `data/catalog/works.csv`, versionado; textos em `data/processed/`,
  fora do Git.
- **Responsável:** proprietário.

## 2026-11-19: Etapa 2, ambiente e convenções de medida

- **PyTorch CPU e NumPy:** dependências do projeto (Apache-2.0/BSD). Localmente usa
  a versão só CPU do PyTorch (índice `pytorch-cpu` no `pyproject.toml`), porque as
  etapas 2 a 4 rodam em CPU; o Colab tem a sua própria versão com GPU.
- **Vocabulário (vocabulary) de caracteres só do treino,** com um símbolo de desconhecido
  (U+FFFD) para caracteres ausentes do treino (1 ocorrência na validação, `½`).
- **Medidas:** perda (loss) em nats por caractere (principal), bits por caractere, bits
  por byte UTF-8 (para comparar com o BPE) e perplexidade (perplexity).
- **Resultados versionados** em `results/` (métricas e amostras, sem pesos);
  tabelas e pesos em `runs/`, fora do Git.
- **Linha de base:** bigrama por contagem, 2,383 nats/caractere (3,33 bits/byte) na
  validação. Detalhes em [docs/etapas/02-bigrama.md](etapas/02-bigrama.md).
- **Responsável:** proprietário.

## 2026-11-19: Repositório criado no GitHub como privado

- **Ação do proprietário:** pediu a criação do repositório no GitHub, **privado**, e
  o envio da `main`.
- **Verificação antes do envio:** busca de padrões sensíveis (e-mails, caminhos da
  máquina, tokens, chaves) nos 12 commits da `main`, sem ocorrências; nenhum dado de
  corpus ou peso no histórico; autor com e-mail `noreply` do GitHub.
- **Continua pendente para torná-lo público:** licença do código, canal privado de
  reporte ativo, secret scanning, push protection, Dependabot e proteção da `main`.
- **Responsável:** proprietário.

## 2026-11-19: Etapa 3, GPT por caracteres

- **Modelo:** Transformer decoder com 4 blocos, 4 cabeças (heads), C = 128, contexto de 128
  caracteres, dropout 0,1; 846.592 parâmetros (parameters). Atenção (attention) escrita explicitamente (Q, K,
  V, máscara causal (causal mask), softmax), sem `scaled_dot_product_attention`.
- **Treino:** loop à mão; AdamW (lr 10⁻³ com warmup de 100 passos e cosseno até
  10⁻⁴, weight decay 0,1 só em matrizes), corte de gradiente em 1, lotes (batches) de 32
  janelas (windows), 3.000 passos, semente (seed) 1337, 11 threads.
- **Motivo do tamanho:** medido nesta CPU (Core Ultra 7 155H, WSL): ~0,6 s por passo
  para este modelo; o modelo da aula do Karpathy (10,9 M) levaria ~10 h. O treino
  maior fica para a GPU (etapa 5).
- **Resultado:** 1,588 nats/caractere (2,22 bits/byte) na validação inteira, contra
  2,383 do bigrama. Checkpoint em `runs/03-gpt-char/` (fora do Git), com SHA-256 em
  `results/03-gpt-char.json`.
- **Responsável:** proprietário.

## 2026-11-20: Etapa 4, tokenizador BPE

- **Tokenizador:** BPE em nível de byte (byte-level), escrito à mão, com
  pré-tokenização (pre-tokenization) por regex (palavra com espaço à frente,
  números de até 3 dígitos, pontuação, espaços); treinado só no treino (teste
  automatizado).
- **Tamanho do vocabulário (vocabulary): 4.096.** Joelho medido: 3,36 bytes por
  símbolo (token) na validação; de 4.096 para 8.192 a compressão melhora 12%, mas as
  junções com menos de 50 ocorrências no treino saltam de 6,5% para 35%.
- **Verificação:** ida e volta (round-trip) exata nos três conjuntos inteiros a cada
  execução; a primeira execução revelou a perda do `_` na pré-tokenização,
  corrigida e coberta por teste.
- **Artefatos:** tokenizador e conjuntos codificados em `runs/04-bpe/` (fora do
  Git), reproduzíveis e conferíveis por SHA-256 em `results/04-bpe.json`.
- **Responsável:** proprietário.

## 2026-11-20: Etapa 5, plano de treino no Colab

- **Ambiente:** Google Colab **gratuito**, GPU T4 (escolha do proprietário). A conta
  Google é escolhida pelo proprietário; o login e o aceite dos termos do Colab são ações
  dele, feitas no navegador. O agente não acessa o Colab nem o Drive.
- **Dados:** o corpus e os símbolos (tokens) são **regenerados no Colab** a partir do commit
  fixado do luxedo, e conferidos contra o Git: o `git diff` do catálogo (SHA-256 do texto de
  cada obra) deve sair vazio, e os SHA-256 do tokenizador e dos três conjuntos devem bater
  com `results/04-bpe.json`. O script de treino recusa dados que não batem. Nada do corpus
  vai para o Drive.
- **Acesso ao código:** o repositório é privado. O Colab o clona com um token
  *fine-grained* do GitHub, só leitura de conteúdo e só deste repositório, com validade
  curta, guardado nos Segredos do Colab (`GITHUB_TOKEN`). O token vai num cabeçalho do
  `git clone`, não fica no `.git/config` nem no notebook. Criar e revogar o token é ação
  do proprietário.
- **Modelos:** pequeno (4 camadas (layers), C = 256, ~4,3 M de parâmetros (parameters))
  e médio (6 camadas, C = 384, ~12 M); 4 e 6 cabeças (heads); contexto (context) de 256
  símbolos; dropout 0,2; matriz compartilhada entre embedding e saída (weight tying). O
  grande (8 camadas, C = 512) só entra se o médio não mostrar sobreajuste (overfitting).
- **Treino:** lotes (batches) de 64 janelas (windows); AdamW com lr 10⁻³ (pequeno) e
  6·10⁻⁴ (médio), warmup de 200 passos e cosseno até 10% do lr; até 5.000 passos; perda
  (loss) exata na validação (validation) inteira a cada 100 passos; parada antecipada
  (early stopping) após 10 avaliações sem melhora; precisão mista (mixed precision)
  float16 com gradient scaler.
- **Checkpoints:** pasta privada no Drive do proprietário: `best.pt` (melhor pela
  validação) e `last.pt` (estado completo, a cada 500 passos, para retomar). SHA-256 do
  `best.pt` em `results/05-gpt-bpe-*.json`. Retomar dá o mesmo resultado que não parar
  (teste automatizado).
- **Escolha e teste:** o modelo é escolhido só pela validação; o teste é reportado uma vez,
  para registro.
- **Pesos (weights):** não publicados (confirmado pelo proprietário).
- **Notebook:** `notebooks/05-colab.ipynb`, versionado **sem saídas (outputs)**; toda a
  lógica fica no pacote, com testes.
- **Responsável:** proprietário.

## 2026-11-23: Token de leitura para o Colab

- **Ação do proprietário:** criou um token *fine-grained* do GitHub
  (`colab-machado-gpt`), só leitura de conteúdo, só deste repositório, validade de 7
  dias, para o Colab clonar o repositório privado. Guardado apenas nos Segredos do Colab.
- **Revogado** pelo proprietário em 2026-11-23, ao fim da etapa 5.
- **Responsável:** proprietário.

## 2026-11-23: Etapa 5, resultado e modelo grande descartado

- **Reprodutibilidade:** no Colab, o corpus, o tokenizador e os três conjuntos saíram
  idênticos, byte a byte, aos locais (catálogo sem diferenças, SHA-256 iguais).
- **Resultado na validação (validation):** pequeno 1,696 e médio 1,663 bits por byte,
  contra 2,22 do GPT por caracteres; teste 1,670 e 1,632. Os dois rodaram os 5.000 passos
  (36 voltas (epochs)) na T4, em 8 e 19 minutos.
- **Modelo grande não treinado:** o médio já mostra sobreajuste (overfitting): distância
  de 0,85 nats entre validação e treino, validação no platô (0,01 nos últimos 1.000
  passos), e só 2% melhor que o pequeno com o triplo de parâmetros (parameters). O
  limite é a quantidade de dados.
- **Checkpoints:** no Drive do proprietário; SHA-256 em `results/05-gpt-bpe-*.json` e no
  diário da etapa.
- **Responsável:** proprietário.

## 2026-11-23: Bot de conversa fica para projeto futuro

- **Decisão do proprietário:** um bot que converse sobre um assunto específico (ideias:
  obra de Machado de Assis, notícias de futebol) fica para um projeto futuro, junto com
  RAG, como o PRD já previa. Este projeto não inclui ajuste por instruções (instruction
  tuning) nem chat.
- **Etapa 6 mantida** como proposta: pré-treino (pretraining) no Wikisource em
  português, sem Machado, e ajuste fino (fine-tuning) com o treino de Machado.
- **Responsável:** proprietário.

## 2026-11-23: Etapa 6, corpus geral do Wikisource

- **Fonte:** dump do Wikisource em português de 2026-11-01
  (`ptwikisource-20261101-pages-articles.xml.bz2`), fixado por data e conferido pelo
  SHA-1 que a Wikimedia publica. Entram as obras digitadas (espaço principal) e os
  livros escaneados, refeitos juntando as páginas "Página:" na ordem.
- **Licença:** os textos do Wikisource são de domínio público ou de licença livre; o
  conteúdo editorial do site (transcrições, traduções feitas por editores) é
  CC BY-SA. O corpus não é versionado nem redistribuído, e os pesos não são
  publicados. O proprietário confirmou (2026-11-23) que isso basta para o critério
  "licença verificada" do PRD, sem verificação obra a obra.
- **Qualidade:** das páginas escaneadas, só as revisadas ao menos uma vez (nível 3 ou
  4). O nível 1 é OCR bruto (8,6 milhões de palavras descartadas), o 2 é marcado como
  problemático e o 0 não tem texto.
- **Dependência nova:** `mwparserfromhell`, só para analisar o wikitexto (parse
  tree); não esconde nada do modelo nem do treino.
- **Machado fora do pré-treino (regra, com testes):** sai todo documento cujo
  cabeçalho ou índice tem Machado de Assis como autor, e todo parágrafo tocado por 2
  ou mais sequências de 8 palavras em comum com o corpus de Machado (treino,
  validação e teste), comparadas com grafia normalizada, para pegar edições em
  ortografia antiga.
- **Validação geral:** ~1% dos documentos, por SHA-256 do título, só para acompanhar o
  pré-treino. A validação e o teste de Machado continuam sendo os da etapa 5.
- **Tokenizador:** o da etapa 4, treinado só no treino de Machado, sem mudança.
- **Responsável:** proprietário (escolha da fonte); critérios técnicos propostos
  pelo agente.

## 2026-11-23: Etapa 6, vazamento de *O Almada* corrigido

- **Achado:** a conferência do corpus já gerado contra a validação e o teste de
  Machado achou *O Almada* (obra da **validação**) dentro do treino do pré-treino, em
  um documento sem autor no cabeçalho e com um verso por parágrafo. O filtro contava
  as sequências de 8 palavras dentro de cada parágrafo, e um verso não tem duas.
- **Correção:** as sequências passam a correr pelo documento inteiro, atravessando
  parágrafos, e cada sequência em comum conta para todo parágrafo que ela toca. Teste
  de regressão em `tests/test_build_general.py`.
- **Resultado:** o filtro por texto foi de 271 parágrafos (31.385 palavras) para 634
  (33.849). Na nova conferência, nenhum parágrafo toca 2 ou mais sequências da
  validação ou do teste de Machado; restam 25 sequências isoladas, todas frases feitas
  ("cada um puxa a brasa para sua sardinha"), nomes ou citações curtas.
- **Corpus final:** 24.623 documentos e 26,7 milhões de palavras (26,2 M no treino,
  0,47 M na validação geral), cerca de 20× o treino de Machado (1,3 M).
- **Responsável:** agente (achado e correção); revisão pelo proprietário.

## 2026-11-23: Etapa 6, plano do pré-treino e do ajuste fino

- **Aprovado pelo proprietário.** Uma variável muda em relação à etapa 5: o modelo
  começa pré-treinado. Modelo, tokenizador, validação de Machado e medida (bits por
  byte) são os mesmos.
- **Pré-treino (pretraining):** modelo médio (~12 M de parâmetros), 1 volta exata pelo
  treino geral (janelas sem sobreposição, embaralhadas e sem repetição, ~3.254
  passos de 16.384 tokens), dropout 0, lr 6·10⁻⁴ com warmup de 200 passos e cosseno
  até 6·10⁻⁵, sem parada antecipada. O `best.pt` é escolhido pela validação geral; a
  validação de Machado é só registrada, como curva.
- **Ajuste fino (fine-tuning):** a partir do `best.pt` do pré-treino, conferido pelo
  SHA-256 registrado; otimizador novo; lr 2·10⁻⁴ (um terço), warmup de 100 passos,
  cosseno até 2·10⁻⁵; dropout 0,2; até 3.000 passos com parada antecipada pela
  validação de Machado (10 avaliações), como na etapa 5. A validação geral é
  registrada para medir o quanto o modelo esquece. O teste é reportado uma vez.
- **Comparação:** contra o médio da etapa 5 (validação 1,663 bpb, teste 1,632). Se o
  ganho for claro, a receita pode ser repetida no pequeno.
- **Ambiente:** Colab gratuito, T4, com dados regenerados lá e conferidos por SHA-256,
  como na etapa 5. Checkpoints no Drive do proprietário. Pesos não publicados.
- **Pendente (ação do proprietário):** novo token *fine-grained* do GitHub, só
  leitura e só deste repositório, de validade curta, para o Colab clonar o
  repositório privado; revogar ao fim da etapa.
- **Responsável:** proprietário.

## 2026-11-24: Etapa 6, resultado

- **Reprodutibilidade:** no Colab, o catálogo de Machado, o tokenizador e os cinco
  conjuntos (três de Machado e dois gerais) saíram idênticos aos locais pelos SHA-256.
  Os dois treinos rodaram no commit `d43f5e7`, na T4.
- **Resultado na validação de Machado:**
  - etapa 5 (médio, só Machado): 1,663 bpb;
  - pré-treinado, sem nunca ver Machado: 1,628 bpb;
  - pré-treinado + ajuste fino: **1,588 bpb** (−4,5%); teste 1,569 bpb (etapa 5:
    1,632).
- **Sobreajuste:** a distância entre validação e treino caiu de 0,85 para 0,34 nats.
- **Esquecimento:** no ajuste fino, a validação geral piorou de 1,557 para 1,803 bpb.
  É esperado, e não afeta o objetivo.
- **Uma execução só**, sem medir a variação entre sementes (seeds).
- **Checkpoints:** no Drive do proprietário; SHA-256 em `results/06-*-medium.json`.
- **Critérios da etapa 6 do PRD cumpridos.**
- **Responsável:** proprietário.

## 2026-11-24: Token de leitura da etapa 6 revogado

- **Ação do proprietário:** criou um token *fine-grained* do GitHub, só leitura de
  conteúdo e só deste repositório, de validade curta, para o Colab clonar o
  repositório privado na etapa 6; guardado apenas nos Segredos do Colab.
- **Revogado** pelo proprietário em 2026-11-24, ao fim da etapa 6.
- **Responsável:** proprietário.
