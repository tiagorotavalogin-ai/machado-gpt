# Etapa 1: Corpus

Diário técnico da etapa 1. Cada seção explica primeiro o conceito, depois o que foi
feito e o que se observou. As decisões formais estão em
[DECISOES.md](../DECISOES.md).

**Objetivo da etapa:** transformar os PDFs da obra de Machado de Assis num corpus
de texto limpo, catalogado e dividido por obra, que servirá de dado de treino nas
próximas etapas.

Por que isso importa para um LLM: o modelo aprende **só** a partir do texto que
recebe. Se o corpus tiver números de página, notas de editores modernos ou palavras
grudadas por erro de extração, o modelo vai aprender a imitar esses defeitos. A
qualidade do dado define o teto do que o modelo pode aprender.

---

## 1. Ambiente Python

### Conceito

Um projeto reproduzível precisa que qualquer pessoa instale **exatamente** as
mesmas versões de bibliotecas. Para isso usamos três peças:

- **`pyproject.toml`:** declara o projeto e as dependências diretas (o que o
  código importa), com versões mínimas.
- **`uv.lock`:** o *lockfile*. Registra a versão exata e o hash de **cada** pacote
  instalado, inclusive as dependências das dependências. Com ele, `uv sync`
  reconstrói o mesmo ambiente em outra máquina.
- **Ambiente virtual (`.venv/`):** uma pasta isolada com o Python e os pacotes do
  projeto, para não misturar com os do sistema. Fica fora do Git.

O código fica em `src/machado_gpt/` (o chamado *src layout*). Isso obriga os testes
a importar o pacote instalado, e não arquivos soltos da pasta, o que evita testes
que só passam na máquina de quem escreveu.

### O que foi feito

```bash
uv sync          # cria .venv/ e instala o que está no uv.lock
uv run pytest    # roda os testes dentro do ambiente
```

Dependências atuais: `pdfplumber` (leitura de PDF) e `pytest` (só desenvolvimento).

---

## 2. Download verificado das fontes

### Conceito: fixar a fonte e provar que o arquivo é o certo

Baixar de uma URL como `.../master/arquivo.pdf` tem dois problemas:

1. **A fonte pode mudar.** `master` aponta para a versão mais recente; se o autor do
   repositório alterar um arquivo, nosso corpus muda sem aviso. Por isso o download
   usa o **commit** `a13ca4a…`, um identificador que nunca muda de conteúdo.
2. **O download pode vir errado.** Conexão cortada, proxy, arquivo trocado. Por isso
   cada arquivo é **verificado por hash**.

Um **hash** é uma "impressão digital" de tamanho fixo calculada a partir dos bytes
de um arquivo. Mudar um único byte muda o hash por completo. Usamos dois:

- **Hash de blob do Git (SHA-1):** é como o Git identifica cada arquivo. Ele calcula
  o SHA-1 de `"blob <tamanho>\0" + conteúdo`. A API do GitHub informa esse hash para
  cada arquivo do commit; se o nosso cálculo bater, o arquivo baixado é
  **exatamente** o daquele commit.
- **SHA-256:** hash mais forte, que vai para o catálogo. É o que usaremos daqui em
  diante para conferir que o corpus não mudou.

### Conceito: tratar o download como dado não confiável

O nome dos arquivos vem de fora. Um nome como `../../.bashrc` faria o script
escrever fora da pasta de dados (ataque chamado *path traversal*). A função
`safe_destination` recusa qualquer caminho absoluto, com `..` ou que, depois de
resolvido, saia de `data/raw/luxedo/`. Os arquivos baixados nunca são executados.

### Outras duas propriedades do script

- **Idempotente:** rodar de novo não baixa o que já está correto. Antes de baixar,
  o script confere o hash do arquivo local.
- **Escrita atômica:** grava primeiro em `arquivo.pdf.part` e só renomeia depois de
  completo. Se o processo cair no meio, não fica um PDF pela metade com o nome final.

### O que foi feito

Código: [`src/machado_gpt/corpus/download.py`](../../src/machado_gpt/corpus/download.py).
Testes: [`tests/test_download.py`](../../tests/test_download.py). Os testes conferem
o hash contra valores conhecidos do `git hash-object`, a recusa de caminhos
perigosos e a codificação de acentos na URL.

```bash
uv run python -m machado_gpt.corpus.download
```

### Resultado

```
232 files at luxedo/dataset-machado-de-assis@a13ca4a
  232/232 verified
manifest: data/catalog/luxedo_manifest.csv (31.0 MB verified)
```

- 116 PDFs (19 MB, a fonte) e 116 TXTs (12 MB, linha de base para comparar a nossa
  extração com a do luxedo), em `data/raw/luxedo/`, fora do Git.
- O catálogo `data/catalog/luxedo_manifest.csv` entra no Git: tem caminho, tamanho,
  hash do Git, SHA-256, repositório e commit de cada arquivo, mas nenhum texto.

---

## 3. Escolha da biblioteca de PDF

A decisão inicial era o PyMuPDF. Antes de instalar, verificamos a licença: ele é
**AGPL-3.0**, uma licença que obrigaria o projeto inteiro a adotar termos
compatíveis, e a licença do código ainda não foi escolhida. Trocamos pelo
**pdfplumber** (MIT), que dá a mesma informação de que precisamos.

Lição: a licença de uma dependência também é uma decisão de projeto, e é mais
barato verificá-la antes de escrever código em cima dela.

---

## 4. Como um PDF guarda texto

### Conceito

Um PDF **não guarda parágrafos nem linhas**. Ele guarda instruções de desenho do
tipo "desenhe o caractere *U* na fonte Verdana, tamanho 9,9, na posição
(98,6; 310,0)". Palavras, linhas e parágrafos são uma **reconstrução** feita por
quem lê o arquivo, a partir de posições e espaçamentos.

Por isso extrair texto de PDF é uma tarefa de inferência, e cada decisão (onde
termina um parágrafo, se um hífen é real, se uma linha é nota do editor) precisa de
um sinal no próprio arquivo. O pdfplumber expõe esses sinais para cada caractere:
texto, fonte, tamanho, `x0` (posição horizontal) e `top` (distância do topo da
página, em pontos; 1 pt = 1/72 de polegada).

### Exemplo: primeira página de *Dom Casmurro*

```
x0=268.9 top= 73.3  9.9 Verdana-Italic | Dom Casmurro
x0=438.6 top=102.0  7.4 Verdana        | Texto de referência:
x0=343.0 top=111.0  7.4 Verdana-Italic | Obras Completas de Machado de Assis, vol. I,
x0=381.8 top=120.0  7.4 Verdana        | Nova Aguilar, Rio de Janeiro, 1994.
x0=262.7 top=143.6  7.4 Verdana        | Publicado originalmente pela Editora Garnier, ...
x0=247.2 top=273.9  9.9 Verdana-Bold   | CAPÍTULO PRIMEIRO
x0=274.4 top=286.0  9.9 Verdana-BoldIt | DO TÍTULO
x0= 98.6 top=310.0  9.9 Verdana        | Uma noite destas, vindo da cidade para o Engenho Novo, ...
x0= 98.6 top=322.0  9.9 Verdana        | Central um rapaz aqui do bairro, que eu conheço de vista ...
...
x0= 98.6 top=382.2  9.9 Verdana        | metesse os versos no bolso.
x0= 98.6 top=406.3  9.9 Verdana        | — Continue, disse eu acordando.
```

O que dá para ler aí:

- **Nota da edição** em fonte 7,4; o texto de Machado está em 9,9. O tamanho da
  fonte separa o que é editorial do que é obra.
- **Títulos de capítulo** em negrito, centralizados (`x0` maior).
- **Parágrafos sem recuo:** todas as linhas começam em `x0=98,6`. O que marca um
  novo parágrafo é o **espaço vertical**: 12 pt entre linhas do mesmo parágrafo
  (322 − 310) e 24 pt entre parágrafos (406 − 382).

---

## 5. Inspeção dos 116 PDFs

### Conceito: medir antes de escrever regras

Um PDF não representa os outros 115. Crítica, poesia e teatro têm formatos
diferentes de romance. Antes de escrever a extração, medimos em **todos** os
arquivos os sinais de que as regras dependem. Assim, cada regra se apoia em dados,
e não em suposições.

Código: [`src/machado_gpt/corpus/inspect_pdfs.py`](../../src/machado_gpt/corpus/inspect_pdfs.py).
Para cada PDF, ele mede páginas sem texto, tamanho da fonte do corpo, porcentagem de
texto em fonte menor, linhas repetidas entre páginas, espaçamentos entre linhas e
linhas terminadas em hífen. O relatório vai para `data/interim/pdf_survey.csv`, fora
do Git.

```bash
uv run python -m machado_gpt.corpus.inspect_pdfs
```

### Resultados

| Gênero | PDFs | Páginas | Linhas terminadas em hífen |
|---|---:|---:|---:|
| conto | 7 | 632 | 257 |
| crítica | 45 | 324 | 74 |
| crônica | 24 | 949 | 320 |
| miscelânea | 10 | 29 | 11 |
| poesia | 7 | 463 | 6 |
| romance | 10 | 1.123 | 425 |
| teatro | 10 | 300 | 25 |
| tradução | 3 | 401 | 138 |
| **total** | **116** | **4.221** | **1.256** |

As traduções foram baixadas, mas ficam fora do corpus principal (ver decisões).

### O que aprendemos

**1. Não é preciso OCR.** Há 30 páginas sem camada de texto, mas todas são a última
página em branco de um PDF (só um traço desenhado). OCR (reconhecimento óptico de
caracteres) só seria necessário em páginas digitalizadas como imagem, e não há
nenhuma.

**2. Repetição não significa cabeçalho.** A regra planejada era "linha que se repete
em muitas páginas é cabeçalho ou rodapé; descarte". A inspeção mostrou que esses
PDFs **não têm** cabeçalho, rodapé nem número de página. As linhas que se repetem
são conteúdo:

- no teatro, os nomes dos personagens antes de cada fala (`MARTE`, `JÚPITER`…);
- nas crônicas, os numerais de seção (`I`, `II`, `III`);
- na poesia, epígrafes e refrões;
- nas críticas, a assinatura "Machado de Assis".

Aplicar a regra apagaria texto verdadeiro. Ela sai.

**3. Sinais de formato são mais confiáveis que palavras.** Todos os 116 PDFs abrem
com a nota da edição moderna, mas o rótulo varia: "Texto-fonte", "Texto-Fonte",
"Textos-Fonte", "Texto de referência", "Edição referência". Nossa primeira busca por
palavras encontrou só 64 de 116; a segunda, 109; só depois de olhar os casos que
faltavam chegamos aos 116. Já o **tamanho da fonte** (7,4 a 7,5 contra 9,9 a 10,0 do
corpo) separou a nota em todos os PDFs que examinamos linha a linha. A extração vai
usar o formato, as palavras ficam como conferência, e um teste vai confirmar a
separação nos 116.

**4. Parágrafos por espaçamento.** Em 108 dos 116 PDFs, as linhas do mesmo parágrafo
ficam a 12 pt umas das outras e os parágrafos a 24 a 26 pt. Os outros 8 usam
outros pares (9/12, 11/22, 10/19…). A regra deve então ser **relativa** ao
espaçamento de linha de cada PDF, e não um número fixo; os 8 casos diferentes serão
conferidos à parte.

**5. Hífen no fim da linha é ambíguo.** Em português, muitos hífens são reais:
"admiti-" + "me" = "admiti-me", e não "admitime". O script do luxedo juntava tudo
("ave-marias" virou "avemarias"). Com 1.256 casos, a regra planejada continua
válida: o próprio corpus decide, verificando se a forma com hífen aparece no meio
de alguma linha.

**6. Notas no fim de algumas obras.** *Americanas* (poesia) tem notas numeradas
(`[1]`, `[2]`…) em fonte 6,0 a 6,5 nas últimas páginas. Na edição original, essas
notas podem ser do **próprio Machado**. **Decisão:** nenhuma nota entra no corpus,
seja do autor ou do editor. Assim não é preciso verificar a autoria nota a nota, e o
corpus fica só com o texto corrido das obras.

**7. Índice no início.** *Poesias dispersas* tem um "ÍNDICE" logo após a nota da
edição. O índice é estrutura, não texto literário, e deve sair.

---

## 6. Extração de um PDF: *Dom Casmurro*

### Conceito: decidir cada fronteira entre linhas

Depois de ler as linhas, a extração toma uma decisão para cada par de linhas
vizinhas: continuam a mesma frase, abrem um novo parágrafo, são um título, ou uma
palavra foi partida por hífen? O código tem duas camadas:

1. **Leitura** (`read_lines`): o pdfplumber transforma o PDF em objetos `Line`, com
   texto, página, posição (`top`, `x0`, `x1`), tamanho da fonte e se a linha está
   toda em negrito.
2. **Regras:** funções puras, que recebem linhas e devolvem decisões. Como não
   dependem de um PDF, cada regra é testada com linhas inventadas em
   [`tests/test_extract.py`](../../tests/test_extract.py).

Código: [`src/machado_gpt/corpus/extract.py`](../../src/machado_gpt/corpus/extract.py).

```bash
uv run python -m machado_gpt.corpus.extract romance/domCasmurro.pdf
```

A saída vai para `data/interim/extracted/` (fora do Git): um `.txt` com parágrafos
separados por linha em branco e um `.json` com a nota da edição (metadado), as
notas separadas e o registro de cada decisão de hífen.

### As regras

| Regra | Sinal usado |
|---|---|
| Nota da edição vira metadado | primeiro bloco contínuo de letra miúda na página 1 |
| Notas saem do corpus | qualquer outra linha em fonte menor que o corpo |
| Título fica isolado | linha inteira em negrito |
| Novo parágrafo na mesma página | espaço vertical maior que 1,5 × o espaçamento de linha |
| Novo parágrafo na quebra de página | ver abaixo |
| Hífen no fim da linha | o próprio texto decide (ver abaixo) |
| Unicode | NFC, ligaduras desfeitas, travessão mantido |

### Um bug que o teste pegou

A primeira versão considerava que o texto da obra começava na primeira linha em
tamanho normal da página 1. Mas o **título** ("Dom Casmurro") vem antes da nota da
edição e está em tamanho normal; com isso, a nota foi classificada como nota comum.
O teste com linhas inventadas falhou antes de olharmos o resultado real. A regra
passou a ser: a nota da edição é o **primeiro bloco contínuo** de letra miúda.

### Hífens: hipótese confirmada

A regra procura a forma com hífen ("admiti-me") ou a forma junta ("admitime") em
outros lugares do texto, sempre **no meio de uma linha**, onde não há dúvida. A
palavra partida não conta como evidência dela mesma.

Resultado em *Dom Casmurro*: 57 linhas terminadas em hífen, 26 confirmadas como
hífen real, 31 sem evidência e **nenhuma** para juntar. As 31 sem evidência são todas
pronomes ("amparar-me", "sacudi-la"), e mantêm o hífen. Isso confirma a hipótese de
que esses PDFs foram gerados a partir de HTML, que não parte palavras no fim da
linha: todo hífen no fim da linha é um hífen de verdade.

### Parágrafos na quebra de página: medir a regra

Dentro de uma página, o espaço vertical diz onde começa um parágrafo. Na passagem
de uma página para outra não há espaço para medir, então é preciso outro sinal. A
primeira ideia era "se a linha termina em pontuação, o parágrafo acabou".

Para saber se a regra é boa, medimos onde a resposta é conhecida: nas fronteiras
**dentro** da mesma página, em que o espaçamento dá a resposta certa.

| Situação da linha anterior | Casos | Encerrou parágrafo |
|---|---:|---:|
| sem pontuação final | 3.747 | 0,1% |
| com pontuação final, linha curta | 1.228 | 100% |
| com pontuação final, linha cheia, próxima começa com travessão | 62 | 100% |
| com pontuação final, linha cheia, sem travessão | 290 | 20% |

"Linha curta" é a que termina antes de 90% da margem direita. A regra da quebra de
página ficou: **novo parágrafo se a linha termina em pontuação e é curta, ou se a
próxima começa com travessão**. Pela tabela, o erro esperado fica restrito ao último
caso (cerca de 1 em 5 dessas quebras); antes, era de cerca de 1 em 6 de **todas** as
quebras com pontuação. Lição: uma regra intuitiva pode ser medida antes de ser
adotada, usando os casos em que a resposta é conhecida.

### Comparação com o `.txt` do luxedo

Comparamos as duas extrações palavra por palavra (`difflib.SequenceMatcher`):

| | Nossa | luxedo |
|---|---:|---:|
| palavras | 66.174 | 66.174 |
| palavras idênticas | 66.100 | 66.100 |
| parágrafos | 1.718 | 1 (quebras de linha do PDF mantidas) |

As 62 diferenças:

- **57 hífens** que o luxedo grudou: "admitime", "fazêlo", "avemarias",
  "tiquetique". A nossa versão mantém "admiti-me", "fazê-lo", "ave-marias".
- **5 diferenças de ordem**: num trecho, a extração do luxedo embaralhou fragmentos
  ("neguei a pés juntos", "era natural que lhe"…) e os jogou para o fim do
  parágrafo. A nossa lê na ordem certa.

Nenhuma palavra do texto se perdeu, e as diferenças são todas a nosso favor. Além
disso, ganhamos os parágrafos, que o luxedo não tinha.

### Limites conhecidos

- Os parágrafos nas quebras de página são uma inferência, e a regra foi medida só
  em *Dom Casmurro*.
- Poesia e teatro ainda não foram testados: versos usam o espaçamento de linha, e a
  regra atual juntaria os versos de uma estrofe num parágrafo.

---

## 7. Romances e contos: as regras se sustentam?

### Conceito: validar fora do caso usado para criar a regra

As regras da seção 6 foram ajustadas olhando *Dom Casmurro*. Uma regra pode
funcionar bem no exemplo em que foi criada e falhar nos outros (em aprendizado de
máquina, isso se chama *overfitting*; aqui vale a mesma ideia). Por isso a aplicamos,
sem mudar nada, aos outros 9 romances e aos 7 contos, que têm o mesmo formato, e
comparamos cada livro com o luxedo.

Para não repetir scripts improvisados, a comparação virou um módulo:
[`src/machado_gpt/corpus/compare.py`](../../src/machado_gpt/corpus/compare.py). Ele
classifica cada diferença em **hífen** (o luxedo grudou: "admitime", ou
"explicou-seme" para "explicou-se-me") ou **outra** (palavras ausentes, a mais ou
fora de ordem).

```bash
uv run python -m machado_gpt.corpus.extract romance/*.pdf conto/*.pdf  # ~3 min
uv run python -m machado_gpt.corpus.compare romance conto
```

(No shell, rode a partir de `data/raw/luxedo/pdf/` para expandir os nomes, ou liste
os arquivos.)

### Resultado

| Obra | Palavras | Iguais ao luxedo | Hífen | Outras |
|---|---:|---:|---:|---:|
| conto/contosFluminenses | 56.589 | 99,90% | 43 | 7 |
| conto/historiasMeiaNoite | 41.817 | 99,89% | 28 | 14 |
| conto/historiasSemData | 50.528 | 99,90% | 45 | 4 |
| conto/paginasRecolhidas | 22.606 | 99,92% | 19 | 0 |
| conto/papeisAvulsos | 54.752 | 99,93% | 38 | 1 |
| conto/reliquias | 29.731 | 99,88% | 25 | 5 |
| conto/variasHistorias | 44.094 | 99,88% | 50 | 2 |
| romance/casaVelha | 22.227 | 99,84% | 35 | 2 |
| romance/domCasmurro | 66.174 | 99,89% | 57 | 5 |
| romance/esau | 72.252 | 99,90% | 29 | 18 |
| romance/helena | 56.247 | 99,89% | 46 | 8 |
| romance/iaia | 56.635 | 99,93% | 34 | 2 |
| romance/maoLuva | 35.300 | 99,86% | 29 | 5 |
| romance/memorial-de-aires | 50.864 | 99,91% | 41 | 5 |
| romance/memoriasBras | 61.267 | 99,80% | 69 | 15 |
| romance/quincas | 77.320 | 99,88% | 49 | 16 |
| romance/ressurreicao | 36.909 | 99,85% | 29 | 11 |
| **total** | **835.312** | | **666** | **120** |

### Lendo as 120 diferenças "outras"

Uma pergunta resolve a maior parte: **as duas versões têm o mesmo conjunto de
palavras?** Se sim, nenhuma palavra se perdeu e a diferença é só de ordem. Contamos
as palavras de cada versão (ignorando hífens) e comparamos as contagens:

- Em 9 dos 17 livros, as contagens são idênticas: as diferenças são só de ordem.
- Nos outros 8, o que sobra são defeitos do luxedo: espaço depois do hífen
  ("casar- se", "bem- aventurança"), uma mensagem de imagem quebrada incluída no
  texto de *Memórias Póstumas* (nós a separamos como nota) e o marcador de cena,
  `* * *` no nosso texto e `***` no luxedo.

**Nenhuma palavra do texto de Machado presente no luxedo falta na nossa extração.**

E quem está certo nas mudanças de ordem? Conferimos um trecho de *Esaú e Jacó*
contra as posições no PDF:

```
pág 79 top=308.8 | Aires quis aquietar-lhe o coração. Nada se mudaria; o regime, sim, era possível,
pág 79 top=320.8 | mas também se muda de roupa sem trocar de pele. Comércio é preciso. Os
```

No PDF, "Nada se mudaria; o regime..." está numa linha só. O luxedo cortou essa linha
em duas e jogou a segunda metade para depois do parágrafo. Isso acontece quando o PDF
guarda uma linha em dois pedaços de texto e o extrator os lê na ordem em que estão
gravados no arquivo. O pdfplumber agrupa os caracteres pela **posição na página**,
que é a ordem de leitura de quem olha a página.

### Sanidade dos parágrafos

São 21.561 parágrafos nos 17 livros. Dois sinais de erro foram procurados:

- **Parágrafo começando com minúscula** (quebra falsa): 12 casos. Os conferidos são
  legítimos: a frase continua depois de um verso citado ("recitei os dois versos…
  *Oh! flor do céu!*… e o outro com grande brio:").
- **Parágrafo gigante** (quebra perdida): o maior, com 5.624 caracteres, é uma carta
  inteira em *Helena*. Os demais ficam abaixo de 2.800.

### O que fica de lição

As regras, criadas com um único livro, se sustentaram em outros 16 sem nenhum
ajuste. Isso aconteceu porque elas se apoiam em sinais medidos (tamanho da fonte,
espaçamento, margem), e não em palavras específicas daquele livro.

---

## 8. Poesia: versos, estrofes e o que não é de Machado

### O que a poesia trouxe de novo

Ao inspecionar os 7 livros de poesia, apareceram coisas que a prosa não tinha:

| Material | Exemplo | De quem é | Destino |
|---|---|---|---|
| Verso e estrofe | uma linha por verso, espaço maior entre estrofes | Machado | fica, com a quebra de linha |
| Índice | "ÍNDICE / MUSA CONSOLATRIX / VISIO…" | estrutura | sai |
| Notas no tamanho do corpo | "[18] Publicado na Biblioteca Brasileira…" | editor moderno | sai |
| Epígrafe | "She speaks! / O speak again… / SHAKESPEARE" | outro autor | sai (decisão do proprietário) |
| Mote repetido | "Voilà ce que l'on dit de moi…" (48 vezes na *Gazeta*) | outro autor | sai |
| Prosa do autor | advertência de *O Almada*, carta-posfácio de *Crisálidas* | Machado | fica |
| Rubrica | "(Estão no fim de um banquete…)" em *Falenas* | Machado | fica |
| Dedicatória e data | "A F. X. DE NOVAIS", "1862" | Machado | fica |
| Poema traduzido ou de outro poeta | "As Ondinas (Noturno de H. Heine)" | outro autor | lista manual (próximo passo) |

### Verso ou prosa? Medir antes de decidir

Dentro de um bloco, as linhas devem ser unidas com **quebra de linha** (verso) ou
com **espaço** (prosa)? Medimos dois sinais em romances e em livros de poesia:

| Sinal no bloco | Romances | Poesia |
|---|---:|---:|
| alguma linha de continuação começa com minúscula | 2.881 | 90 |
| nenhuma começa com minúscula, mas as linhas vão até a margem | 51 | 74 |
| nenhuma começa com minúscula e as linhas são curtas | 10 | 3.073 |

A última linha da tabela é verso: em poesia, quase todos os blocos; em romance, os
10 casos são versos citados ("Chora, menina, chora, / Chora, porque não tem /
Vintém,"), que antes eram unidos com espaço por engano. A regra ficou: **verso é o
bloco em que nenhuma linha de continuação começa com minúscula e as linhas não
vão até a margem.** Versos longos (alexandrinos em *Ocidentais*) não quebram de
linha no PDF: conferimos que a linha seguinte sempre começa com maiúscula.

### Estrofes na quebra de página: a pontuação engana

Na prosa, "termina em pontuação e é curta" acertou 100% das quebras de parágrafo.
Em verso, toda linha é curta, e a pontuação sozinha é um sinal fraco: dentro da
página, só **42%** dos versos com pontuação final fecham estrofe (2.387 de 5.661).
Usar a regra da prosa erraria cerca de 75 das 370 quebras de página.

O sinal melhor é a **regularidade**. Muitos poemas têm estrofes de tamanho fixo
(quadras, oitavas). Se as estrofes inteiras de um poema têm 4 versos e, na quebra
de página, já há 4 versos antes, a estrofe acabou; se há 2 antes e 2 depois, ela
continua. Medimos que essa regra decide 192 das 288 estrofes que cruzam páginas
(todas as 102 da *Gazeta*). Nos outros casos, a estrofe continua, porque dentro da
página isso é o mais comum (98% depois de verso sem pontuação, 58% depois de verso
com pontuação).

Um cuidado: num romance, duas falas curtas de diálogo em volta de uma quebra de
página parecem uma estrofe. Por isso a regra de verso só vale em livros que são
majoritariamente verso, medido pelos trechos inteiros dentro de uma página.

### Epígrafes: uma regra lapidada por erros

A regra de epígrafe é a que mais mudou neste passo. Cada versão foi aplicada aos
24 livros, e cada erro encontrado virou um teste:

| Versão | Erro encontrado | Correção |
|---|---|---|
| linha recuada logo após o título | removia meios-versos de Machado ("Herói lhe chamam") | exigir atribuição no fim |
| atribuição em maiúsculas | títulos de capítulo centralizados viraram "epígrafe" | parar em linha em negrito |
| idem | removia o epitáfio de *Memórias Póstumas* ("AQUI JAZ…") | exigir texto em caixa mista |
| idem | rubrica "(…Três escravos.)" e dedicatórias de álbum | atribuição precisa ter palavra em maiúsculas; rubrica começa com "(" |
| idem | "A misérrima Dido" tomada por dedicatória | dedicatória com "A" precisa nomear alguém (maiúscula, inicial, "Sr.") |
| mote por semelhança | datas de cartas ("Corte, 5 de outubro") viraram mote | mote precisa de 2 linhas ou mais |
| sequência pode cruzar a página | uma cena dialogada inteira (57 linhas) foi removida | epígrafe tem no máximo 10 linhas (a maior real tem 8) |

A versão final reconhece a epígrafe assim: um trecho logo depois de um título,
recuado para a direita (em poesia, alinhado num recuo fixo; em prosa, alinhado à
direita), de até 10 linhas, com texto em caixa mista, que termina numa atribuição
(SHAKESPEARE; "CAMÕES — Elegias"; em prosa, um nome curto como "Dante") ou se
repete em 3 ou mais poemas (o mote da *Gazeta*, reconhecido mesmo com variações
de digitação). Datas e dedicatórias no começo do trecho ficam no texto; "DO AUTOR"
(Machado citando a si mesmo) também fica.

Resultado: 90 epígrafes removidas (85 na poesia, 5 na prosa: Jeremias, Molière,
Montaigne, Diderot, Dante). Todas estão no `.json` de cada obra, para conferência.

**Lição:** uma heurística parece certa no exemplo em que foi pensada. O que a
torna confiável é rodá-la em tudo, ler o que ela removeu e transformar cada erro em
teste. Os testes garantem que uma correção não desfaz a anterior.

### Notas e marcadores

- **Notas no corpo:** a seção de notas começa na primeira linha "[n] texto" e vai até
  o fim do livro (*O Almada*, *Poesias dispersas*). Um "[52]" sozinho na linha é só
  o marcador, não uma nota.
- **Marcadores:** "[12]" e "[iii]" saem do texto ("CLEÓPATRA[iii]" vira
  "CLEÓPATRA").
- **Inserções do editor:** linhas inteiras entre colchetes ("[Fragmento de III]")
  saem.
- **Índices:** saem todos, inclusive o índice de capítulos de cada conto; o índice
  termina onde a primeira entrada aparece de novo, como título de verdade.

### Conferência: alguma palavra sumiu sem explicação?

Para cada livro, pegamos as palavras do luxedo que não estão no nosso texto e
descontamos as que estão no material removido de propósito (cabeçalho, notas,
índice, epígrafes). O que sobra é perda não explicada:

- **67 de 952.069 palavras**, todas identificadas: espaço depois do hífen no luxedo
  ("bem- aventurança"), a mensagem de imagem quebrada em *Memórias Póstumas*,
  `***` contra `* * *` e números de página de um índice.
- *Crisálidas* mostrou um caso real, já corrigido: os marcadores em romanos
  ("[iii]") grudavam nos títulos.

### Resultado

| Gênero | Palavras | Blocos | Blocos em verso | Epígrafes removidas |
|---|---:|---:|---:|---:|
| romance (10) | 535.188 | 13.198 | 22 | 1 |
| conto (7) | 299.525 | 8.146 | 29 | 4 |
| poesia (7) | 109.332 | 4.005 | 3.033 | 85 |

Nos romances, as únicas mudanças em relação ao passo anterior são títulos em
maiúsculas separados do texto (antes, alguns grudavam no primeiro parágrafo),
versos citados com quebra de linha e a epígrafe de Dante em *Esaú e Jacó*. A
regra de estrofe na quebra de página não alterou nenhum bloco de prosa.

### Pendente para o próximo passo

Uma **lista manual com fonte** de trechos de outros autores que as regras não
detectam com segurança:

- poemas traduzidos por Machado (por exemplo, "As Ondinas", de Heine; "O Corvo", de
  Poe; "Ser ou não ser", de Shakespeare);
- poemas de outros poetas incluídos nos livros (o posfácio de *Crisálidas* menciona
  versos de amigos);
- 2 epígrafes na margem normal em *Americanas* (Simão de Vasconcelos; "MAT., XIII,
  3."), que não têm recuo.

---

## 9. Lista manual: traduções e textos de outros autores

### Conceito: quando a regra automática não basta, curadoria com evidência

Algumas coisas não dá para detectar com segurança pelo formato da página. "O Corvo"
parece um poema de Machado como qualquer outro; só o subtítulo "(EDGAR POE)" diz
que é tradução. Para esses casos, usamos uma **lista manual**, com três exigências:

1. **Evidência para cada item:** onde está escrito que o texto não é de Machado.
2. **Verificação automática:** cada item aponta para um trecho exato do texto
   extraído; se o trecho não for encontrado, ou for encontrado mais de uma vez, a
   extração **para com erro**, em vez de seguir em silêncio.
3. **Dúvidas separadas:** o que não é claro fica marcado como `revisar` até o
   proprietário decidir.

### A melhor fonte estava no próprio material

As notas de rodapé dos livros de poesia, que já tínhamos separado do corpus, são do
**próprio Machado**, das edições originais. Ele declara as traduções com todas as
letras:

- "Já bastava o arrojo de traduzir a maviosa elegia de Chenier" (*Crisálidas*);
- "o belo canto de que fiz esta mal amanhada paráfrase" (Mme. de Girardin);
- "Não é original esta composição; o original é propriamente indígena" (*Americanas*);
- "Não sei alemão; traduzi estes versos pela tradução em prosa francesa" (Schiller).

A outra fonte são os **subtítulos** dos poemas: "(EDGAR POE)", "(Noturno de H.
Heine)", "(Imitação de Cowper)". Para ligar cada nota ao seu poema, procuramos no
PDF a linha que carrega o marcador (por exemplo, `CLEÓPATRA[iii]`).

As duas fontes são primárias e verificáveis em cada PDF, então não foi preciso
recorrer a fontes externas. A única identificação feita por nós é o original de *A
Morte de Ofélia* (subtítulo "Paráfrase"; o conteúdo corresponde a *Hamlet*, ato IV,
cena 7), e isso está anotado no catálogo.

### O catálogo

O arquivo é [`data/catalog/exclusions.csv`](../../data/catalog/exclusions.csv) e é
versionado. Cada linha diz:

| Coluna | Significado |
|---|---|
| `obra` | livro (`poesia/ocidentais`) |
| `inicio` | começo do primeiro bloco removido (precisa ser único) |
| `fim_antes_de` | começo do bloco onde o texto mantido recomeça (vazio: até o fim) |
| `tipo` | `traducao`, `imitacao`, `outro_autor`, `epigrafe`, `nota_do_autor`, `lingua_estrangeira` |
| `autor_original`, `fonte` | quem escreveu e onde está a evidência |
| `acao` | `excluir`, ou `revisar` (fica no texto até a decisão) |

O código está em
[`src/machado_gpt/corpus/exclusions.py`](../../src/machado_gpt/corpus/exclusions.py)
e é aplicado no fim da extração. O `.json` de cada obra registra o que foi removido,
com o número de blocos e de palavras de cada trecho; conferimos um a um que cada
trecho tem o tamanho de um poema.

Para encontrar o fim de cada trecho, não confiamos numa regra: um detector de
"próximo título" parou em dedicatórias ("A ANÍSIO SEMPRÔNIO RUFO"), em seções
internas ("I - UM PASSEIO DE CARRO") e em "FIM". Cada limite foi conferido contra
o índice do livro.

### O que saiu

| Tipo | Itens | Exemplos |
|---|---:|---|
| tradução | 20 | Musset, Chénier, Heine, Schiller, Poe, Shakespeare, La Fontaine, Dante; a *Lira Chinesa* (8 poemas) conta como um item |
| poema ou trecho de outro autor | 2 | *Embirração*, resposta de F. X. de Novais; a ode de Anacreonte na tradução de Castilho, citada dentro de um poema dramático de Machado (o resto do poema fica) |
| epígrafe sem recuo | 3 | Simão de Vasconcelos, Mateus, C. A. de Sá |
| notas de Machado no corpo | 1 | notas de 1864 no fim de *Crisálidas* |
| decisão do proprietário | 4 | *Un vieux pays* e *Refus* (de Machado, em francês); *Minha Mãe* e *O Casamento do Diabo* (imitações) |

São 30 itens e cerca de 11.600 palavras. A poesia passou de 109.332 para **97.740
palavras**. Romances e contos não têm itens na lista.

### Lição

Uma lista manual não é um atalho, desde que cada item tenha evidência e o código
verifique cada item. Ela é o complemento honesto das regras: as regras cobrem o que
o formato revela, e a lista cobre o que só o conteúdo revela.

---

## 10. Teatro: falas, rubricas e um erro antigo revelado

### Dois formatos de fala

As 10 peças usam dois formatos:

| Formato | Exemplo no PDF | Peças |
|---|---|---|
| nome numa linha, fala abaixo (separados por 24 pt) | `VALENTIM` / `Ah! És tu?` | *O Caminho da Porta*, *Desencantos*, *Os Deuses de Casaca*… |
| nome e fala na mesma linha | `D. HELENA (imitando-a) — Helena!` | *Lição de Botânica*, *As Forcas Caudinas*… |

Os dois problemas que a extração tinha:

1. **No formato em linha**, as falas ficam a 12 pt umas das outras, como linhas de
   um mesmo parágrafo, e várias falas eram unidas num bloco só. Regra nova: uma
   linha que começa com `NOME —` abre bloco (verificamos que, fora do teatro, o
   padrão só aparece em falas de cenas dentro de contos e em títulos).
2. **No formato em duas linhas**, o nome ficava num bloco sozinho, separado da fala.
   Regra nova, só para teatro: o nome é unido à fala por uma quebra de linha,
   absorvendo as rubricas que estiverem no meio. "Nome de personagem" é um bloco
   curto em maiúsculas que aparece sozinho pelo menos 3 vezes na peça; títulos
   como "ATO ÚNICO" aparecem uma vez e ficam como estão.

Resultado, nos dois formatos, um bloco por fala:

```
VALENTIM
(irônico)
Ah!

SEABRA (fechando o livro) — É melhor. As coisas boas não se gozam de uma assentada.
```

Não inventamos nenhum caractere: o nome continua numa linha própria, como no PDF.

### Um erro antigo: o espaçamento de linha

Ao conferir as peças, uma carta de Machado com vários parágrafos saiu como um bloco
só. A causa estava na regra mais antiga da extração: o espaçamento de linha era "a
distância mais comum entre linhas". Numa peça, as falas curtas separadas por 24 pt
são tantas que 24 pt vira a distância mais comum. Com isso, o limite de parágrafo
subia para 36 pt, e os parágrafos normais (24 pt) eram unidos.

Medimos um sinal mais robusto: a distância **depois de uma linha cheia**, que só
pode ser seguida pela continuação do mesmo parágrafo.

| Livros | Moda geral | Moda após linha cheia |
|---|---:|---:|
| 24 romances, contos e livros de poesia | 12 | 12 |
| 9 das 10 peças | **24** | 12 |

Nos outros gêneros as duas medidas coincidem, então a correção não mudou nenhum
livro fora do teatro (conferido bloco a bloco). **Lição:** uma regra que funcionou
em 24 livros ainda pode estar errada; o erro só aparece quando os dados mudam de
perfil.

### Outro erro antigo: onde há espaço entre palavras

A conferência contra o luxedo mostrou, pela primeira vez, palavras que **nós**
erramos: "Nãome", "dadocomo" (grudadas) e "P ERSONAGENS", "CA MÕES" (partidas).
Olhando caractere por caractere:

- letras da mesma palavra ficam a ~0 pt umas das outras;
- espaços reais ficam entre ~2,6 e 3,5 pt, mas o pdfplumber só via espaço a partir
  de 3 pt (`x_tolerance=3`), e grudava as palavras com espaço de 2,6 pt;
- em alguns trechos, o PDF tem um caractere de espaço gravado **dentro** da palavra,
  sobreposto às letras.

Correção: ignorar os espaços gravados e deduzir o espaço só pela distância, com
tolerância de 1,5 pt (no meio entre ~0 e ~2,6). Mudaram só 3 peças, todas para
melhor ("Não me", "Tão longe", "PERSONAGENS"), e as palavras sem explicação na
comparação com o luxedo, somando todos os gêneros, caíram de 90 para 75 (todas
defeitos já identificados do luxedo).

### Verso no teatro

*O Bote de Rapé* parecia ter blocos demais em verso para uma comédia. Conferimos: é
uma **comédia em verso**, em alexandrinos ("Preciso escolher a cambraia, / A renda,
o gorgorão e os galões para a saia"). A regra de verso acertou. *Os Deuses de
Casaca* também é em verso.

### Lista manual: 4 itens novos

| Peça | Trecho | Motivo |
|---|---|---|
| *Hoje avental, amanhã luva* | a peça inteira | "Comédia em um ato **imitada do francês**" (mesma decisão das imitações) |
| *O Caminho da Porta* | "CARTA AO AUTOR" | carta de Quintino Bocaiúva em resposta à de Machado |
| *Os Deuses de Casaca* | "NOTA" final | nota de Machado no corpo do texto |
| *Tu, só tu, puro amor* | epígrafe de Camões | título em caixa mista; a regra automática não detecta |

Ficam: a carta de Machado a Quintino Bocaiúva, a "NOTA PRELIMINAR" de *Quase
Ministro* e a advertência de *Tu, só tu, puro amor*, que são prefácios do autor
sobre as peças, e não notas ao texto.

### Resultado

| Gênero | Obras | Palavras | Blocos |
|---|---:|---:|---:|
| romance | 10 | 535.188 | 13.198 |
| conto | 7 | 299.525 | 8.146 |
| poesia | 7 | 97.740 | 3.592 |
| teatro | 10 (1 excluída inteira) | 59.316 | 4.352 |
| **total** | **34** | **991.769** | |

---

## 11. Crônica, crítica e miscelânea: as regras gerais se sustentam

### Conceito: rodar primeiro, corrigir depois

São 79 PDFs. Em vez de inspecionar um a um, aplicamos as regras como estavam e
medimos: comparação com o luxedo, tamanho dos blocos e blocos em verso. Só olhamos
de perto os livros que fugiram do padrão. Em 74 dos 79 PDFs, faltavam no máximo 5
palavras em relação ao luxedo, e nenhum bloco ficou gigante.

### Letras espaçadas: a tolerância de espaço, de novo

Dois textos (*O Jornal e o Livro*, *A Queda que as Mulheres têm para os Tolos*) têm
trechos com **letras espaçadas**, a ~1,5 pt umas das outras. A tolerância de 1,5 pt
escolhida no teatro partia essas palavras ("desespe rado"). O luxedo é pior nesses
trechos ("Pigm a l i ã o").

Desta vez medimos a distribuição das distâncias entre letras em **todos** os PDFs:

| Distância entre letras | O que é |
|---|---|
| abaixo de 1 pt | letras da mesma palavra (quase tudo) |
| 1,4 a 1,6 pt | letras da mesma palavra, em trecho espaçado |
| 2,0 pt | o menor espaço real entre palavras |
| acima de 3,1 pt | espaços entre palavras (quase tudo) |

A tolerância passou para **1,8 pt**, no meio do vale entre 1,6 e 2,0. Também
passamos a ignorar o espaço não separável (`\xa0`) gravado como caractere. Na
reextração dos 113 PDFs, só os dois textos espaçados mudaram, e para melhor.

**Lição:** a tolerância de 1,5 pt foi escolhida olhando um caso ("dadocomo"); a de
1,8 pt, olhando a distribuição inteira. Um número tirado de um exemplo vale até o
próximo exemplo; um número tirado da distribuição explica por que funciona.

### Comentários do editor dentro do texto

*Pareceres do Conservatório Dramático* vem de manuscritos, e o editor moderno anotou
a transcrição dentro das linhas:

- comentários saem inteiros: "[ilegível]", "[pouco legível: recusa?]", "[a palavra
  março está riscada…]", "[?]";
- palavras restauradas pelo editor ficam, sem colchetes: "vendido [no] México" vira
  "vendido no México".

A limpeza é feita no parágrafo já montado, porque um comentário pode atravessar a
quebra de linha.

### Citações em verso: ficam (decisão do proprietário)

Nas críticas, Machado cita os poetas que comenta em blocos de verso; nas crônicas,
cita outros poetas e às vezes escreve versos próprios. São cerca de 13.600 palavras
(1,9% desses gêneros). Diferente da epígrafe, a citação faz parte do argumento
("O poeta assiste à saída de Tereza: / Olhem como vem pimpona!"), e removê-la
deixaria frases penduradas. Citações em prosa, dentro das frases, não teriam como
sair de qualquer forma. **Decisão:** as citações ficam e são registradas como limite
conhecido do corpus: ele é de Machado, mas inclui trechos que ele citou.

### Textos de outros autores

Procuramos respostas, cartas recebidas e traduções. Tudo o que apareceu é de Machado:
cartas dele ("Carta ao Sr. Conselheiro J. F. de Castilho"), pareceres dele sobre
traduções alheias e menções a cartas de outros que não estão reproduzidas ("Acabo de
ver uma carta dirigida pelo… Sr. Bitencourt da Silva"). Nenhum item novo na lista
manual.

### O corpus completo

| Gênero | Obras | Palavras | Blocos |
|---|---:|---:|---:|
| romance | 10 | 535.188 | 13.198 |
| conto | 7 | 299.525 | 8.146 |
| poesia | 7 | 97.740 | 3.592 |
| teatro | 10 (1 excluída inteira) | 59.316 | 4.352 |
| crônica | 24 | 539.260 | 10.422 |
| crítica | 45 | 161.981 | 2.655 |
| miscelânea | 10 | 11.653 | 214 |
| **total** | **113** | **1.704.663** | **42.579** |

As traduções (3 PDFs) foram baixadas, mas não são extraídas (decisão de não incluir
traduções).

**Conferência final contra o luxedo:** 205 palavras do luxedo não aparecem no nosso
texto nem no material removido de propósito. Cerca de 100 são os pedaços das
palavras espaçadas do luxedo ("Pigm", "a", "l", "i", "ã", "o"); o resto segue o
padrão já identificado (espaço depois de hífen, `***`, a mensagem de imagem quebrada).

---

## 12. Catálogo, divisão por obra e estatísticas

### Conceito: por que dividir por obra

Treinamos o modelo num conjunto (**treino**) e medimos a qualidade em textos que
ele nunca viu: **validação** (usada durante o desenvolvimento, para escolher
hiperparâmetros e comparar versões) e **teste** (guardado para a medição final).
Se um parágrafo de *Dom Casmurro* estiver no treino e o seguinte na validação, o
modelo já viu os personagens, as frases e o assunto, e a validação passa a medir
memória, não generalização. Por isso a divisão é feita **por obra inteira**.

E não basta separar arquivos: a mesma obra pode estar em dois arquivos.

### Procurando a mesma obra em dois arquivos

Para cada par de obras, contamos as **sequências de 8 palavras seguidas** em comum.
Frases de 8 palavras raramente se repetem por acaso; muitas em comum indicam texto
repetido.

| Par | Parte da menor contida na outra | O que é |
|---|---:|---|
| *Crítica Teatral* ↔ *Revista Dramática* | 98,6% | coletânea dos mesmos folhetins |
| *A Mãe* ↔ *Revista Dramática* | 94,0% | idem |
| *Entre 1892 e 1894* ↔ *A Semana* | 70,0% | crônicas publicadas nas duas coleções |
| *Contos Fluminenses* ↔ *As Forcas Caudinas* | 27,2% | a peça reescrita como conto: duas versões |
| 4 pares de poesia com prosa ou outra poesia | até 5,6% | poemas de Machado republicados em outro livro |

Duas consequências:

- **Grupos:** obras com 50 ou mais sequências em comum formam um grupo, e o grupo
  inteiro vai para o mesmo conjunto (o agrupamento é transitivo).
- **Duplicatas:** uma obra com 90% ou mais do texto dentro de outra sai do corpus
  (*Crítica Teatral* e *A Mãe*), porque o texto repetido pesaria em dobro no treino.
  Fica a obra maior; nenhuma palavra se perde.

### Conceito: aleatório, mas reproduzível

A ordem das obras para o sorteio é dada pelo **SHA-256** de `semente:nome do grupo`.
Um hash espalha os nomes de modo que parece aleatório, mas é sempre igual: qualquer
pessoa, em qualquer máquina, obtém a mesma divisão, que não depende da ordem dos
arquivos nem de um gerador de números aleatórios. Trocar a semente gera outra
divisão, igualmente reproduzível.

### A regra de divisão

Por **gênero**, para que cada conjunto tenha romance, poesia, crônica...:

1. percorrer os grupos na ordem do hash e preencher teste e validação até ~10% das
   palavras do gênero cada; um grupo que passaria de 125% do alvo vai para o treino
   (por isso *A Semana*, com 16% do corpus sozinha, fica no treino);
2. se um conjunto ficar vazio num gênero, ele recebe o primeiro grupo, na ordem do
   hash, com até 25% do gênero.

A segunda regra nasceu de uma medição: na primeira versão, **a poesia ficou inteira
no treino** (os livros têm 15 a 20 mil palavras, acima do limite de 12 mil) e o
conto ficou sem validação. Sem poesia na validação, não saberíamos como o modelo se
sai em verso.

### Resultado

| Gênero | Treino | Validação | Teste |
|---|---:|---:|---:|
| romance | 419.805 (7 obras) | 56.247 (1) | 59.136 (2) |
| conto | 226.545 (5) | 50.415 (1) | 22.565 (1) |
| poesia | 69.257 (5) | 15.299 (1) | 13.184 (1) |
| teatro | 48.321 (7) | 5.948 (1) | 5.047 (1) |
| crônica | 414.719 (8) | 68.170 (10) | 56.371 (6) |
| crítica | 112.851 (24) | 15.985 (9) | 15.754 (10) |
| miscelânea | 9.330 (7) | 1.182 (1) | 1.141 (2) |
| **palavras** | **1.300.828 (77%)** | **213.246 (13%)** | **173.198 (10%)** |
| **caracteres** | 7.463.240 | 1.245.299 | 997.937 |

São 110 obras no corpus (113 extraídas, menos 2 duplicatas e *Hoje avental, amanhã
luva*, excluída pela lista manual), com cerca de 1,69 milhão de palavras e 9,7
milhões de caracteres.

**Conferência nos arquivos finais:** das sequências de 8 palavras da validação,
0,014% também aparecem no treino; no teste, 0,013%. São frases comuns ("entrou a
andar de um lado para outro") e uma citação da *Ilíada* que Machado usa em crônicas
diferentes, e não texto repetido de uma obra.

### Arquivos

- [`data/catalog/works.csv`](../../data/catalog/works.csv) (versionado): uma linha
  por obra, com gênero, conjunto, situação, grupo, palavras, caracteres, edição de
  referência, fonte, licença, SHA-256 do PDF e do texto limpo.
- `data/processed/{train,val,test}.txt` (fora do Git): o texto de cada conjunto.
  Parágrafos e estrofes são separados por uma linha em branco; obras, por duas.
- Código: [`src/machado_gpt/corpus/build.py`](../../src/machado_gpt/corpus/build.py).
  O teste `test_catalog_has_no_leakage_between_splits` lê o catálogo real e falha se
  um grupo estiver em dois conjuntos.

---

## Etapa 1: critérios de pronto (PRD)

| Critério | Situação |
|---|---|
| Catálogo com fonte, edição, licença e hash de cada texto | `data/catalog/works.csv` |
| Textos limpos, sem notas ou prefácios de editores modernos | seções 6 a 11 |
| Divisão por obra reproduzível (hash determinístico) | seção 12, com teste de vazamento |
| Estatísticas do corpus | seção 12 |

**Limites conhecidos, registrados:** citações em verso de outros poetas dentro de
críticas e crônicas ficam no corpus; parágrafos e estrofes nas quebras de página são
inferidos; a lista manual de exclusões cobre o que encontramos, não tudo o que pode
existir.

## Como reproduzir esta etapa até aqui

```bash
uv sync
uv run pytest                                    # 82 testes
uv run python -m machado_gpt.corpus.download     # baixa e verifica (~31 MB)
uv run python -m machado_gpt.corpus.inspect_pdfs # inspeção (~6,5 min na CPU)
uv run python -m machado_gpt.corpus.extract romance/domCasmurro.pdf  # ou várias obras (todas: ~6 min)
uv run python -m machado_gpt.corpus.compare romance conto
uv run python -m machado_gpt.corpus.build                # catálogo e divisão
```

## Próximo passo

**Etapa 2: modelo bigrama.** O modelo mais simples possível: prever o próximo
caractere olhando só o caractere atual. Primeiro por contagem (uma tabela), depois
como uma rede em PyTorch que aprende a mesma tabela por gradiente. A perda de
validação dele será a linha de base que todo modelo seguinte precisa superar.
