# Etapa 6: Pré-treino geral e ajuste fino com Machado

Diário técnico da etapa 6. Cada seção explica o conceito, depois o que foi feito, o
resultado medido e a lição. As decisões formais estão em
[DECISOES.md](../DECISOES.md).

**Objetivo da etapa:** pré-treinar (pretraining) o modelo num corpus geral em
português, sem Machado, ajustá-lo (fine-tuning) com o treino de Machado e comparar
com o modelo "só Machado" da etapa 5, na mesma validação (validation).

**Estado:** corpus geral pronto e tokenizado. Pré-treino e ajuste fino ainda não
rodaram.

---

## 1. Por que pré-treinar

A etapa 5 terminou num limite de **dados**, não de modelo. O médio (12 milhões de
parâmetros (parameters)) já decorava o treino: 0,85 nats de distância entre
validação e treino, validação parada nos últimos 1.000 passos, e só 2% melhor que
o pequeno com o triplo de parâmetros. Com 2,25 milhões de tokens, 5.000 passos dão
36 voltas (epochs) pelo mesmo texto.

Boa parte do que o modelo precisa aprender não é "Machado", é **português**: que
"que" vem antes de verbo, como se concorda o adjetivo, onde vai a vírgula. Isso se
aprende com qualquer texto. A ideia do pré-treino é:

1. **Pré-treino:** aprender português num corpus grande e variado, sem Machado.
2. **Ajuste fino:** continuar o treino, com os mesmos pesos (weights), só no treino
   de Machado, para aprender o estilo dele.

É o que os modelos grandes fazem, em escala muito maior. A pergunta desta etapa é
se, nesta escala pequena, isso já melhora a validação de Machado.

---

## 2. A fonte: Wikisource em português

O Wikisource é a biblioteca de textos da Wikimedia: obras em domínio público ou de
licença livre, transcritas por voluntários. Usamos o **dump** (cópia completa do
site) de 2026-11-01, fixado pela data e conferido pelo SHA-1 que a Wikimedia
publica: rodar de novo dá o mesmo corpus.

Há dois tipos de página com texto:

- **Obras digitadas** (espaço principal): o texto está na própria página.
- **Livros escaneados** (espaço "Página:"): cada página do livro é transcrita à
  parte, ao lado da imagem. A página da obra só as inclui (transclusão). Para ter o
  livro, juntamos as páginas na ordem do número.

### Wikitexto para texto

As páginas estão em wikitexto, a marcação do MediaWiki (`{{c|CAPÍTULO III}}`,
`[[Autor:Fulano|Fulano]]`, `<ref>nota</ref>`). Usamos a biblioteca
`mwparserfromhell` só para analisar a marcação em árvore (parse tree). O que fazer
com cada nó está escrito à mão em
[`wikisource.py`](../../src/machado_gpt/wikisource.py):

- modelos (templates) de formatação mantêm o texto (`{{c|CAPÍTULO III}}` vira
  "CAPÍTULO III");
- navegação, tabelas, notas (`<ref>`), cabeçalhos de página e caixas de licença
  saem;
- a palavra hifenizada na virada de página (`{{hífen|aper|aperfeiçoamento}}` no fim
  de uma, o resto no começo da outra) aparece uma vez só, inteira;
- como no MediaWiki, uma quebra de linha simples é um espaço e uma linha em branco
  separa parágrafos; poemas (`<poem>`) mantêm os versos.

### Qualidade das páginas escaneadas

Cada página escaneada tem um nível de revisão. Ficamos só com as **revisadas ao
menos uma vez** (níveis 3 e 4):

| nível | significado | palavras |
|---|---|---|
| 0 | sem texto | 4.315 |
| 1 | OCR bruto, não revisado | 8.638.700 (descartadas) |
| 2 | marcada como problemática | 48.204 (descartadas) |
| 3 | revisada | 7.932.438 |
| 4 | validada por uma segunda pessoa | 1.215.828 |

O nível 1 é quase metade dos escaneados. Ensinar o modelo com erros de OCR
("rn" no lugar de "m", palavras cortadas) seria ensinar português errado.

---

## 3. Tirar Machado do pré-treino (regra, com testes)

Se o pré-treino lesse a validação ou o teste de Machado, a comparação com a etapa 5
seria trapaça: o modelo já teria visto as respostas. Também o tiramos do **treino**
de Machado, para que ele só entre no ajuste fino e a separação entre as duas fases
fique limpa.

Dois filtros:

1. **Por autor:** sai o documento cujo cabeçalho (ou o índice do livro escaneado)
   tem Machado de Assis como autor. Foram 1.175 documentos, 1,24 milhão de
   palavras.
2. **Por texto:** Machado também aparece sem o nome, citado em antologias, em
   jornais, em documentos sem cabeçalho. Sai todo parágrafo que tenha **2 ou mais
   sequências de 8 palavras** em comum com o corpus de Machado.

### Grafia antiga

O corpus de Machado da etapa 1 está em grafia modernizada; muitas edições do
Wikisource estão na original ("ella", "hontem", "theatro", "commercio", "Braz").
Uma comparação literal não acharia nada. Antes de comparar, cada palavra é
normalizada: minúsculas, sem acentos, ph→f, th→t, y→i, z→s, letras dobradas
simples, h mudo fora (mantendo ch, lh, nh). Assim, "Ella sahiu hontem do theatro"
e "Ela saiu ontem do teatro" viram a mesma sequência.

### Por que 2 sequências, e não 1

Uma sequência de 8 palavras sozinha pode ser frase feita, que qualquer autor
escreve ("cada um puxa a brasa para sua sardinha", "da Lei de 28 de setembro de
1871"). Duas sequências em comum já indicam o mesmo trecho.

### O vazamento de *O Almada*

Depois de gerar o corpus, conferimos o resultado em vez de confiar no filtro:
procuramos no corpus geral todas as sequências de 8 palavras da validação e do
teste de Machado.

Apareceu *O Almada*, poema de Machado que está na **validação**, inteiro, cerca de
1.850 palavras no treino do pré-treino. Estava num documento sem autor no
cabeçalho, com **um verso por parágrafo**:

```
A Preguiça, no entanto, conduzira

Aos macios colchões o grande Almada,

E um sono amigo lhe fechara os olhos,
```

O filtro contava as sequências **dentro** de cada parágrafo. Um verso tem 8 a 10
palavras, então tem no máximo 2 ou 3 sequências próprias, e quase nunca 2 em comum.
Cada verso escapava sozinho.

**Correção:** as sequências passam a correr pelo documento inteiro, atravessando os
parágrafos, e cada sequência em comum conta para todo parágrafo que ela toca. Uma
frase feita no meio de outro texto continua tocando 1 sequência só. Versos seguidos
de um poema de Machado formam muitas sequências, e todos saem. Um teste de
regressão reproduz o caso com versos de 5 palavras.

| | parágrafos removidos por texto | palavras |
|---|---|---|
| filtro por parágrafo | 271 | 31.385 |
| filtro pelo documento | 634 | 33.849 |

Na nova conferência, nenhum parágrafo toca 2 ou mais sequências da validação ou do
teste. Restam 25 sequências isoladas, todas frases feitas, nomes próprios ou
citações curtas.

### Lições

- **Um filtro é uma hipótese sobre os dados.** "Um trecho de Machado tem 2
  sequências no mesmo parágrafo" era verdade para prosa e falso para verso. Só a
  conferência do resultado mostrou isso.
- **Conferir no resultado, não no código.** Os testes do filtro passavam; o
  vazamento só apareceu quando procuramos a validação dentro do corpus gerado.
- **Um caso achado vira teste.** O teste de regressão garante que uma mudança
  futura não volte ao filtro antigo.

---

## 4. Validação geral

Separamos ~1% dos documentos, pelo SHA-256 do título (o mesmo método da divisão por
obra da etapa 1), como **validação geral**. Ela serve só para acompanhar o
pré-treino: saber se o modelo está aprendendo português ou decorando o Wikisource.
A medida que importa no fim continua sendo a validação de Machado da etapa 5.

---

## 5. O corpus geral

Código: [`build_general.py`](../../src/machado_gpt/build_general.py). Resultado em
[`results/06-wikisource.json`](../../results/06-wikisource.json).

| | |
|---|---|
| documentos | 24.623 |
| palavras | 26,7 milhões (26,2 M treino, 0,47 M validação geral) |
| digitadas / escaneadas | 17,8 M / 8,9 M palavras |
| em relação ao treino de Machado (1,3 M palavras) | ~20× |

Os autores com mais texto, depois dos documentos sem autor no cabeçalho (4,8 M
palavras) e das coletâneas ("Vários"), são Eça de
Queirós, Monteiro Lobato, Aluísio Azevedo, Joaquim Manuel de Macedo e Lima Barreto:
boa parte do século XIX, próximo da época de Machado.

---

## 6. Tokenização

O tokenizador é o **da etapa 4, sem mudança**, treinado só no treino de Machado.
Treinar um novo no corpus geral talvez comprimisse melhor, mas mudaria duas coisas
de uma vez em relação à etapa 5, e a comparação deixaria de dizer qual delas
ajudou.

Código: [`tokenize_general.py`](../../src/machado_gpt/tokenize_general.py).

| | tokens | bytes por token |
|---|---|---|
| treino geral | 53.316.135 | 3,01 |
| validação geral | 905.201 | 3,25 |
| treino de Machado (etapa 4) | 2.251.901 | 3,42 |

O texto geral comprime menos: o tokenizador aprendeu as junções (merges) com o
vocabulário de Machado, e o Wikisource tem grafia antiga, legislação e outros
assuntos. É o preço esperado de manter o tokenizador.

Com o lote (batch) da etapa 5 (64 × 256 = 16.384 tokens por passo), 5.000 passos
leem 82 milhões de tokens: **1,5 volta** pelo corpus geral, contra 36 pelo de
Machado. O modelo verá quase tudo uma vez só, e decorar fica bem mais difícil.

### Dois problemas achados no caminho

1. **Memória.** A primeira versão abria 21 processos e juntava os ids numa lista de
   inteiros Python (cerca de 28 bytes cada, contra 2 de um `uint16`), numa máquina
   com 3 GB livres. Foi a causa provável de o editor ter caído no meio da
   tokenização. A versão atual usa no máximo 6 processos, devolve `uint16` e
   confere a ida e volta (round trip) de cada documento no próprio processo. Pico
   de 1,4 GB; 1 minuto no total.
2. **Separador entre documentos.** Codificar documento por documento só dá os
   mesmos ids que codificar o texto inteiro se a pré-tokenização (pre-tokenization)
   cortar no mesmo lugar. No texto inteiro, `\n\n\nPalavra` vira `\n\n` + `\n` +
   `Palavra`: a regra de espaços deixa o último antes da palavra. Um pedaço que
   **termina** em `\n\n\n` vira um único `\n\n\n`. Pondo o separador no **começo**
   do documento seguinte, as duas formas coincidem. O teste que comparava as duas
   achou a diferença; a ida e volta exata, sozinha, não acharia, porque as duas
   formas decodificam no mesmo texto.

---

## 7. Arquivos e reprodução

```bash
uv run python -m machado_gpt.build_general     # baixa o dump (~1x), monta o corpus (~12 min)
uv run python -m machado_gpt.tokenize_general  # codifica (~1 min)
uv run pytest
```

- Corpus em `data/processed/general/` e tokens em `runs/06-general/`, fora do Git.
- Estatísticas, SHA-256 do tokenizador e dos `.bin` em
  `results/06-wikisource.json`, versionado.
- Testes: [`test_wikisource.py`](../../tests/test_wikisource.py),
  [`test_build_general.py`](../../tests/test_build_general.py),
  [`test_tokenize_general.py`](../../tests/test_tokenize_general.py).

---

## 8. Plano do treino (aprovado)

Uma variável muda em relação à etapa 5: o modelo começa pré-treinado. Modelo,
tokenizador, validação e medida (bits por byte) continuam os mesmos.

### Fase 1: pré-treino

- **Modelo:** o **médio** da etapa 5 (6 camadas (layers), C = 384, ~12 M de
  parâmetros). É o que mais sofreu com a falta de dados.
- **Dados:** o treino geral, 53,3 M de tokens.
- **Duração:** **1 volta**, 3.254 passos de 16.384 tokens. Cada trecho é visto uma
  vez só, então não há como decorar.
- **Uma volta de verdade.** Na etapa 5, cada lote (batch) sorteia janelas em
  posições aleatórias, **com reposição**. Sorteando tantas janelas quanto cabem no
  texto, cerca de 1/e (37%) do texto nunca é lido e outra parte é lida duas vezes.
  No Machado, com 36 voltas, isso não importava. Numa volta só, importa. O
  pré-treino corta o texto em janelas sem sobreposição, embaralha e serve cada uma
  uma vez (`WindowSampler`). O ajuste fino continua sorteando, como na etapa 5.
- **Dropout 0.** O dropout existe contra o sobreajuste (overfitting), que não
  acontece numa volta só. Desligá-lo deixa o modelo aprender mais por passo. É a
  prática dos pré-treinos de verdade. O dropout não tem pesos, então pode voltar a
  0,2 no ajuste fino sem mudar nada no modelo.
- **Taxa de aprendizado (learning rate):** a do médio na etapa 5, 6·10⁻⁴, com
  warmup de 200 passos e cosseno até 6·10⁻⁵.
- **O que acompanhar:**
  - a validação geral, que diz se o modelo está aprendendo português;
  - a **validação de Machado**, só como curva. Ela mostra quanto o modelo "sabe de
    Machado" sem nunca tê-lo lido.
- **Escolha:** o `best.pt` do pré-treino é o de menor perda na validação **geral**
  (numa volta só, na prática é o do fim). A validação de Machado não decide nada
  nesta fase.

### Fase 2: ajuste fino (fine-tuning)

- **Partida:** os pesos do fim do pré-treino, com um otimizador (AdamW) novo.
- **Dados:** o treino de Machado, o mesmo da etapa 5.
- **Taxa de aprendizado menor, 2·10⁻⁴** (um terço da do pré-treino), com warmup de
  100 passos e cosseno até 2·10⁻⁵. Uma taxa grande desmancharia o que o pré-treino
  ensinou; o modelo só precisa se ajustar.
- **Dropout 0,2**, como na etapa 5: agora o texto se repete e o sobreajuste volta.
- **Até 3.000 passos**, com parada antecipada (early stopping) pela validação de
  Machado (10 avaliações sem melhora), como na etapa 5. O teste é reportado uma vez.

### A comparação

| | validação | teste | distância validação − treino |
|---|---|---|---|
| etapa 5, médio só Machado | 1,663 bpb | 1,632 bpb | 0,85 nats |
| pré-treinado, antes do ajuste | ? | | |
| pré-treinado + ajuste fino | ? | ? | ? |

Mais as amostras (samples) com os mesmos começos da etapa 5 ("Capitu", "Não
consultes dicionários."). Se o ganho for claro, a mesma receita pode ser repetida no
pequeno, para ver se a vantagem do tamanho reaparece quando há dados.

### Custo e ambiente

- **Colab gratuito, T4,** como na etapa 5. Tempo estimado: cerca de 13 min de
  pré-treino e até 12 min de ajuste fino.
- **Dados regenerados no Colab,** conferidos pelos SHA-256 de
  `results/06-wikisource.json` e `results/04-bpe.json`: o dump tem 88 MB, e montar e
  tokenizar o corpus leva cerca de 20 min.
- **Checkpoints** no Drive do proprietário, como na etapa 5.

### Código

- **Treino:** `train()` da etapa 5 ganha dois parâmetros opcionais: pesos de
  partida e conjuntos extras para avaliar. Sem eles, o comportamento é o de antes,
  e os testes da etapa 5 continuam valendo.
- **Script novo** `train_general.py`, com `--phase pretrain` e `--phase finetune`.
  Ele confere os dados pelos SHA-256 e grava `results/06-*.json`.
- **Testes:**
  - o ajuste fino parte exatamente dos pesos do pré-treino;
  - o pré-treino recusa dados que não batem com os SHA-256;
  - retomar dá o mesmo resultado que não parar.
- **Notebook** `notebooks/06-colab.ipynb`, versionado sem saídas (outputs).

---

## Próximo passo

Aprovar o plano, implementar o código com testes, rodar um teste curto (smoke run)
local e depois treinar no Colab.
