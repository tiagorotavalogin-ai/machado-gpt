# Etapa 2: Modelo bigrama

Diário técnico da etapa 2. Cada seção explica o conceito, depois o que foi feito, o
resultado medido e a lição. As decisões formais estão em
[DECISOES.md](../DECISOES.md).

**Objetivo da etapa:** construir o modelo de linguagem mais simples possível, medir
sua perda na validação e usá-la como **linha de base**: todo modelo seguinte só
vale a pena se ficar abaixo dela.

---

## 1. O que é um modelo de linguagem

Um modelo de linguagem atribui uma probabilidade ao **próximo símbolo**, dado o
que veio antes. Para gerar texto, sorteamos o próximo símbolo dessa distribuição,
acrescentamos ao texto e repetimos.

O **bigrama** é o caso mínimo: o "antes" é só o símbolo atual. Nesta etapa, o
símbolo é o **caractere**. O modelo responde a uma pergunta: depois de um "q", qual
a probabilidade de vir "u"? E de vir "a"?

---

## 2. Vocabulário: só do treino

Cada caractere distinto do **treino** recebe um número (id). A regra do projeto é
que nada da validação ou do teste influencie o modelo, nem pelo vocabulário. Por
isso, um caractere que nunca aparece no treino vira um símbolo único de
"desconhecido" (`�`, U+FFFD).

| | |
|---|---|
| caracteres distintos no treino | 143 |
| + símbolo desconhecido | **V = 144** |
| caracteres da validação ausentes do treino | 1 ocorrência (`½`) |
| caracteres do teste ausentes do treino | nenhum |

Código: [`src/machado_gpt/vocab.py`](../../src/machado_gpt/vocab.py), com testes de
ida e volta (codificar e decodificar devolve o texto) e do símbolo desconhecido.

---

## 3. Como medir: a perda

A **perda** (*loss*) é a média de −log P(caractere que realmente veio) sobre todas
as posições da validação. Se o modelo dá probabilidade 1 ao caractere certo, −log 1
= 0; se dá probabilidade baixa, a perda cresce. Ela aparece em quatro unidades:

| Unidade | O que é |
|---|---|
| **nats por caractere** | perda com logaritmo natural; é o que o PyTorch calcula |
| **bits por caractere** | a mesma perda com log₂: quantos bits, em média, um compressor ideal gastaria por caractere usando o modelo |
| **bits por byte** | total de bits dividido pelo número de bytes UTF-8 ("ç" ocupa 2 bytes); é a medida que permite comparar com o tokenizador BPE da etapa 4, cujos símbolos não são caracteres |
| **perplexidade** | exp(perda): "entre quantos caracteres o modelo hesita, em média" |

Para saber se o modelo aprendeu algo, duas linhas de base:

- **uniforme:** todos os 144 caracteres igualmente prováveis; perda = ln 144 =
  4,970 nats;
- **unigrama:** cada caractere com a sua frequência no treino, sem olhar o anterior.

---

## 4. Bigrama por contagem

### Conceito

Para estimar P(b | a), contamos quantas vezes `b` veio depois de `a` no treino e
dividimos pelo total de vezes que `a` apareceu. O modelo inteiro é uma tabela de
144 × 144 = 20.736 números.

Um problema: um par que nunca aparece no treino tem probabilidade 0, e −log 0 é
infinito. Se ele aparecer na validação, a perda explode. A correção é a
**suavização**: somar α a todas as contagens antes de dividir, e escolher α na
validação.

Código: [`src/machado_gpt/bigram.py`](../../src/machado_gpt/bigram.py).

```bash
uv run python -m machado_gpt.bigram
```

### Resultado

| Modelo | nats/caractere | bits/byte | perplexidade |
|---|---:|---:|---:|
| uniforme | 4,970 | 6,95 | 144,0 |
| unigrama | 3,109 | 4,35 | 22,4 |
| **bigrama por contagem (α = 0,1)** | **2,383** | **3,33** | **10,8** |

- **Olhar um caractere para trás reduz a hesitação** de ~22 para ~11 caracteres.
- **A suavização quase não importa aqui:** 84% dos pares (17.379 de 20.736) nunca
  aparecem no treino, mas também quase não aparecem na validação. Qualquer α entre
  0,001 e 1 dá praticamente a mesma perda (2,383 a 2,384).
- **Treino 2,373, validação 2,383:** quase iguais. Com 7,4 milhões de caracteres, uma
  tabela de 20 mil números não tem como decorar o texto. Não há *overfitting*.

Amostra (começando em "Capitu", semente 1337):

> Capituempo Gomené ssodintiramivim, quao ncitomintre impartumatititos fo J. O
> qumamorgoraçõe dera, hor de arado

O modelo acerta as combinações de duas letras do português ("qu", "çõe", "nt"),
mas não forma palavras: não lembra nada além do último caractere.

---

## 5. O mesmo bigrama como rede neural

### Conceito

Em vez de contar, guardamos uma matriz **W** de 144 × 144 números livres
(*logits*), começando em zero. Para o caractere atual `a`:

1. pegamos a linha `W[a]` (o mesmo que multiplicar o vetor *one-hot* de `a` por W);
2. aplicamos o **softmax**, exp(W[a]) dividido pela soma, e temos uma distribuição
   de probabilidade;
3. a perda é −log P(caractere real): a **entropia cruzada**;
4. o **gradiente** da perda diz como mexer cada número de W para a perda cair;
   damos um passo pequeno nessa direção: `W -= lr * W.grad`.

O passo de atualização é escrito à mão, sem a classe pronta de otimizador, para
deixar o mecanismo visível. Cada passo usa um **lote** de 32.768 posições sorteadas
do treino (descida de gradiente estocástica).

Este modelo tem exatamente a mesma capacidade da tabela de contagem: o melhor W
possível é o logaritmo das probabilidades contadas. **Se o treino funcionar, ele
deve chegar à mesma perda.** Esse é o teste do loop de treino antes de usá-lo no
GPT.

Um detalhe de eficiência: a perda de validação de uma tabela só depende de quantas
vezes cada par aparece. Por isso ela é calculada como soma(contagem × −log P) sobre a
tabela 144 × 144, em vez de passar 1,2 milhão de posições pela rede. Um teste
confirma que as duas contas dão o mesmo número.

### Resultado

| Passo | Treino | Validação |
|---:|---:|---:|
| 0 | 4,970 | 4,970 |
| 300 | 2,451 | 2,457 |
| 900 | 2,405 | 2,412 |
| 1.800 | 2,391 | 2,400 |
| 3.000 | 2,385 | **2,394** |

O passo 0 é exatamente a uniforme (W = 0). Depois de 3.000 passos (~2,5 min em 22
núcleos de CPU), a validação chega a **2,394 nats**, 0,011 acima da contagem
(2,383), e a curva ainda está descendo. Um teste com texto sintético confirma que o
mesmo loop recupera a tabela de contagem quando treinado o suficiente.

### O que a amostra revelou

A amostra neural tem um lixo que a da contagem não tem: "lan+“2.", "pó86W=Jj".
Hipótese: as linhas de W dos **caracteres raros** quase não recebem gradiente,
porque aparecem pouco nos lotes, e ficam perto do ponto de partida (uniforme).
Medimos, por faixa de frequência do caractere atual:

| Caractere atual | Linhas | Entropia aprendida (uniforme = 4,97) | Distância à contagem (KL) | Parte da diferença de perda |
|---|---:|---:|---:|---:|
| mais de 100 mil ocorrências | 16 | 2,26 | 0,004 | +0,0029 |
| 10 mil a 100 mil | 27 | 1,94 | 0,042 | +0,0032 |
| mil a 10 mil | 33 | 2,58 | 0,257 | +0,0029 |
| menos de mil | 68 | **4,89** | **1,181** | +0,0018 |

A hipótese se confirma, com uma nuance:

- **Os caracteres raros quase não aprenderam:** entropia 4,89, colada na uniforme.
  Depois de um deles, o modelo sorteia quase qualquer caractere; daí o lixo.
- **Mas eles pesam pouco na perda,** porque também são raros na validação. A maior
  parte da diferença vem das linhas frequentes, que ainda estavam convergindo.

**Lição:** a perda média e a qualidade da amostra contam histórias diferentes. A
perda pesa cada caso pela frequência; a amostra expõe os casos raros. É preciso
olhar as duas.

---

## 6. Linha de base registrada

| Modelo | Validação (nats/caractere) | bits/byte |
|---|---:|---:|
| bigrama por contagem | **2,383** | **3,33** |
| bigrama neural (3.000 passos) | 2,394 | 3,35 |

O GPT por caracteres da etapa 3 precisa ficar **abaixo de 2,383 nats/caractere** na
mesma validação. Os números completos, com as amostras, estão em
[`results/02-bigram.json`](../../results/02-bigram.json) (versionado). As tabelas
treinadas ficam em `runs/02-bigram/`, fora do Git.

## Como reproduzir

```bash
uv sync
uv run pytest                        # 95 testes
uv run python -m machado_gpt.bigram  # contagem + rede (~2,5 min em CPU)
```

Pré-requisito: o corpus da etapa 1 em `data/processed/` (ver
[01-corpus.md](01-corpus.md)).

## Etapa 2: critérios de pronto (PRD)

| Critério | Situação |
|---|---|
| Modelo de contagem e versão em PyTorch | seções 4 e 5 |
| Loss de validação registrada como linha de base | seção 6 e `results/02-bigram.json` |
| Amostras geradas | seções 4 e 5 e `results/02-bigram.json` |

## Próximo passo

**Etapa 3: GPT por caracteres.** O bigrama esquece tudo além do último caractere. O
próximo modelo olha um **contexto** de muitos caracteres, e o mecanismo que permite
isso é a **atenção**: cada posição decide quais posições anteriores importam para
prever a próxima.
