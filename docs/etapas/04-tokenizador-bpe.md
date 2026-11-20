# Etapa 4: Tokenizador BPE

Diário técnico da etapa 4. Cada seção explica o conceito, depois o que foi feito, o
resultado medido e a lição. As decisões formais estão em
[DECISOES.md](../DECISOES.md).

**Objetivo da etapa:** trocar o caractere por um símbolo (token) maior, aprendido
dos dados, com um tokenizador (tokenizer) BPE treinado só no conjunto de treino
(training), que reproduza qualquer texto exatamente e com um tamanho de
vocabulário (vocabulary) justificado por medição.

---

## 1. Por que sair dos caracteres

No GPT por caracteres, cada posição do contexto (context) é uma letra. O modelo
gasta capacidade **soletrando**: "Capitu, " ocupa 8 posições, e o contexto de 128
posições cobre pouco mais de uma linha de texto.

Se sequências frequentes ("que", " de", "ção") fossem um símbolo só, o mesmo
contexto cobriria mais texto e cada posição carregaria mais significado. É o que
os GPTs de verdade fazem.

---

## 2. Como o BPE funciona

**BPE** (*Byte Pair Encoding*) aprende o vocabulário a partir do texto:

1. **Partir dos bytes.** O texto vira os seus bytes UTF-8, e o vocabulário inicial
   são os 256 valores possíveis de um byte. Por isso o BPE "em nível de byte"
   (*byte-level*) representa **qualquer** texto, mesmo caracteres que nunca
   apareceram no treino (o `½` da validação, um emoji): não existe símbolo
   "desconhecido".
2. **Juntar o par mais frequente.** Contamos todos os pares de símbolos vizinhos no
   treino e criamos um símbolo novo para o par mais frequente (*merge*, junção).
   Os bytes de " " e "d" viram o símbolo 256.
3. **Repetir** até o vocabulário ter o tamanho desejado. A lista ordenada de
   junções **é** o tokenizador.

Para **codificar** (encode) um texto, aplicamos as junções na ordem em que foram
aprendidas. Para **decodificar** (decode), juntamos os bytes de cada símbolo e
lemos como UTF-8.

### Pré-tokenização

Antes de contar pares, o texto é cortado em pedaços (pré-tokenização,
*pre-tokenization*): palavra com o espaço da frente (" que"), números em grupos de
até 3 dígitos, sequências de pontuação e espaços em branco. As junções nunca
atravessam esses limites. Sem isso, o BPE criaria símbolos como "a, e" ou "o. —",
que misturam palavra e pontuação e desperdiçam o vocabulário. Um teste verifica
que nenhum símbolo aprendido atravessa um limite.

### Escrito à mão

Código: [`src/machado_gpt/bpe.py`](../../src/machado_gpt/bpe.py), sem a biblioteca
`tokenizers` da Hugging Face, para mostrar o mecanismo. Uma otimização: depois de
cada junção, só os pedaços que continham o par são recontados (atualização
incremental). Assim, aprender as 16.128 junções do estudo de tamanhos, sobre 1,7
milhão de pedaços, levou ~3 minutos nesta CPU, e o script completo com V = 4.096
(treino do BPE mais a codificação dos três conjuntos), ~2 minutos.

---

## 3. As duas regras, testadas

### Só no treino

Se o tokenizador visse a validação (validation), os símbolos favoreceriam aquele
texto e a medição ficaria otimista. O teste
[`tests/test_tokenize_corpus.py`](../../tests/test_tokenize_corpus.py) roda o
script com textos falsos e verifica que **o único texto passado ao treino do BPE é
o do treino**; validação e teste só são codificados.

### Ida e volta exata (round-trip)

`decodificar(codificar(texto)) == texto`, byte a byte. Testado de três formas:

- textos aleatórios com acentos, emojis e caracteres de vários alfabetos, que nunca
  apareceram no treino;
- todos os caracteres de U+0020 a U+2FFF, um a um, na pré-tokenização;
- **os três conjuntos inteiros** do corpus, no script, a cada execução.

### Um erro que só o corpus inteiro revelou

A primeira execução parou com "round trip failed on train", apesar de todos os
testes passarem. O `_` (sublinhado) é um caractere de "palavra" (`\w`) para o `re`
do Python, mas não é letra nem dígito; nenhuma alternativa do padrão de
pré-tokenização o capturava, e ele **sumia** do texto (121 ocorrências no treino).
Os textos de teste simplesmente não tinham `_`.

Correção: o `_` passou a ser tratado como símbolo, e um teste novo varre milhares
de caracteres. **Lição:** testes com exemplos escolhidos verificam o que
imaginamos; a verificação sobre o dado inteiro pega o que não imaginamos. As duas
são necessárias.

---

## 4. Que tamanho de vocabulário?

### O dilema

- **Vocabulário maior:** cada símbolo cobre mais bytes, e o contexto cobre mais
  texto.
- **Vocabulário menor:** cada símbolo aparece mais vezes no treino, e o vetor
  (embedding) dele é bem aprendido; além disso, cada símbolo custa uma linha na
  embedding e uma na camada (layer) de saída.

### A medição

As junções são aprendidas em ordem, então as primeiras N junções de um tokenizador
grande **são** o tokenizador de tamanho 256 + N. Treinamos uma vez até 16.384 e
medimos todos os tamanhos menores:

| V | bytes por símbolo (validação) | texto em 128 símbolos | junções com < 50 ocorrências no treino | mediana de ocorrências por junção |
|---:|---:|---:|---:|---:|
| 256 (só bytes) | 1,00 | 128 bytes | n/a | n/a |
| 512 | 2,03 | 260 bytes | 0,0% | 6.292 |
| 1.024 | 2,52 | 322 bytes | 0,9% | 1.552 |
| 2.048 | 2,95 | 377 bytes | 2,0% | 500 |
| **4.096** | **3,36** | **430 bytes** | **6,5%** | **178** |
| 8.192 | 3,76 | 481 bytes | 35,1% | 66 |
| 16.384 | 4,10 | 525 bytes | 75,4% | 23 |

Dos 256 bytes iniciais, ~137 nunca aparecem em texto português UTF-8 (caracteres
de controle e bytes inválidos no UTF-8); por isso a tabela olha só as junções.

### A escolha: V = 4.096

É o **joelho da curva**:

- até 4.096, cada dobra do vocabulário melhora bem a compressão (de 1,00 para 3,36
  bytes por símbolo);
- de 4.096 para 8.192, a compressão melhora só 12% (3,36 para 3,76), mas as junções
  raras saltam de 6,5% para 35%. Com ~2,2 milhões de símbolos de treino, um terço
  do vocabulário teria vetores mal aprendidos; vimos esse efeito no bigrama, com os
  caracteres raros;
- o custo em parâmetros (parameters) é aceitável: com C = 384, embedding e camada
  de saída somam 3,1 M, ou metade se compartilharem a matriz.

---

## 5. Resultado

```bash
uv run python -m machado_gpt.tokenize_corpus           # V = 4.096 (~2 min)
uv run python -m machado_gpt.tokenize_corpus --sweep   # mais o estudo de tamanhos (~4 min)
```

| Conjunto | Símbolos | Bytes por símbolo | Ida e volta |
|---|---:|---:|---|
| treino | 2.251.901 | 3,42 | exata |
| validação | 382.461 | 3,36 | exata |
| teste | 306.518 | 3,36 | exata |

A compressão é um pouco melhor no treino (3,42) que na validação (3,36): o
tokenizador foi feito para o treino. Essa pequena diferença é o motivo de não
deixá-lo ver a validação.

Exemplo (83 bytes em 22 símbolos):

> `Não` · ` consul` · `tes` · ` d` · `ic` · `ion` · `ários` · `.` · ` Cas` · `m` ·
> `ur` · `ro` · ` não` · ` está` · ` aqui` · ` no` · ` sentido` · ` que` · ` eles` ·
> ` lhe` · ` dão` · `.`

Palavras frequentes são um símbolo só; palavras raras viram pedaços. Os maiores
símbolos aprendidos são palavras longas e frequentes (" acontecimentos",
" circunstâncias", " principalmente") e, curiosamente, sequências de pontos: as
linhas pontilhadas de algumas obras, como o capítulo de diálogo feito só de
reticências em *Memórias Póstumas*.

A ordem em que as junções surgem mostra o BPE descobrindo o português pela
frequência: começos de palavra (" d", " a", " e") e sílabas ("ra", "os") nas
primeiras; palavras curtas (" do") perto da 50ª; palavras inteiras (" sendo") perto
da 2.000ª.

### Arquivos

- [`results/04-bpe.json`](../../results/04-bpe.json) (versionado): estudo de
  tamanhos, contagens, exemplo e SHA-256 do tokenizador e dos arquivos codificados.
- `runs/04-bpe/` (fora do Git): o tokenizador (`bpe-4096.json`, a lista de
  junções) e os três conjuntos codificados (`train.bin`, `val.bin`, `test.bin`, 2
  bytes por símbolo), prontos para a etapa 5. O treino é determinístico: rodar de
  novo gera o mesmo tokenizador, conferível pelo SHA-256.

---

## Etapa 4: critérios de pronto (PRD)

| Critério | Situação |
|---|---|
| Byte-level BPE treinado só no treino | seção 3, com teste |
| Round-trip exato em UTF-8 testado | seção 3: testes e os três conjuntos inteiros |
| Tamanho de vocabulário justificado | seção 4 |

## Próximo passo

**Etapa 5: treino e avaliação no Colab.** Um GPT maior, com o tokenizador BPE,
treinado na GPU do Google Colab, com checkpoints verificados por SHA-256. A
comparação com o GPT por caracteres será em **bits por byte**, a única medida
comum: os símbolos agora são de tamanhos diferentes, e perda (loss) por símbolo de
um e perda por caractere do outro não são comparáveis diretamente.
