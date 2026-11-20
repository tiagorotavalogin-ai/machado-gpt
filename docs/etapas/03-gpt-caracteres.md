# Etapa 3: GPT por caracteres

Diário técnico da etapa 3. Cada seção explica o conceito, depois o que foi feito, o
resultado medido e a lição. As decisões formais estão em
[DECISOES.md](../DECISOES.md).

**Objetivo da etapa:** construir um Transformer (a arquitetura do GPT) peça por
peça, treiná-lo por caracteres e ficar **abaixo da linha de base (baseline) do bigrama**:
2,383 nats por caractere na validação (validation).

---

## 1. O problema que o Transformer resolve

O bigrama prevê o próximo caractere olhando só o atual. Depois de "Capit", ele só
sabe que veio um "t", e por isso não forma palavras. Um modelo melhor precisa
olhar um **contexto**: os últimos T caracteres (aqui, T = 128).

A dificuldade é **como** combinar 128 caracteres. Uma tabela, como a do bigrama,
teria 144¹²⁸ linhas. O Transformer resolve com a **atenção** (attention): cada posição aprende
a consultar as posições anteriores que importam para ela e a juntar o que
encontrou.

---

## 2. As peças, na ordem em que o dado passa

Código: [`src/machado_gpt/gpt.py`](../../src/machado_gpt/gpt.py). Cada peça está
comentada no código.

### Embeddings: o que é e onde está

Cada caractere vira um vetor de C = 128 números, aprendido (a **embedding de
token**). Para a mesma letra, o vetor de partida é o mesmo; o significado no
contexto vem depois, das camadas (layers).

A atenção, sozinha, não sabe a **ordem** das posições: ela trata o contexto como
um conjunto. Por isso somamos uma segunda embedding, a **de posição**: um vetor
aprendido para cada posição 0, 1, ..., 127.

### Autoatenção causal: as posições conversam

Cada posição produz três vetores a partir do seu:

- **consulta** (q, *query*): o que eu procuro;
- **chave** (k, *key*): o que eu ofereço;
- **valor** (v, *value*): o que eu passo adiante.

O peso que a posição i dá à posição j é o produto q_i · k_j, normalizado por
softmax entre todas as posições j. A posição i recebe a média dos valores,
ponderada por esses pesos (weights).

Dois detalhes que importam:

- **Causal:** a posição i só pode olhar para j ≤ i. Um modelo que vê o caractere que
  deve prever não aprende nada útil. Uma **máscara triangular** (causal mask) coloca −∞ nos
  pesos das posições futuras antes do softmax, o que as zera. Um teste verifica:
  mudar um caractere futuro não altera nenhuma previsão anterior.
- **Escala:** o produto q · k é dividido por √(tamanho da cabeça). Sem isso, os
  produtos crescem com a dimensão, o softmax satura e a posição passa a olhar uma
  única outra posição desde o início, o que atrapalha o aprendizado.

### Várias cabeças

Em vez de uma atenção com vetores de 128, usamos **4 cabeças** (heads) com vetores de 32,
em paralelo, e concatenamos os resultados. Cada cabeça pode aprender a procurar
uma coisa diferente: o caractere anterior, o começo da palavra, a abertura das
aspas.

### MLP: cada posição pensa sozinha

Depois de "conversar", cada posição passa por uma pequena rede de duas camadas
(alarga para 4 × 128, aplica uma não linearidade (nonlinearity) GELU e volta para 128). A
atenção mistura informação **entre** posições; a MLP transforma a informação
**dentro** de cada posição.

### Conexões residuais e LayerNorm

Cada sub-bloco **soma** a sua contribuição ao vetor que recebeu: `x + f(x)`, em vez
de substituí-lo por `f(x)`. Isso cria um caminho direto da saída até a entrada,
por onde o gradiente passa sem enfraquecer. É o que permite empilhar muitos
blocos.

Antes de cada sub-bloco, a **LayerNorm** normaliza o vetor (média 0, variância 1,
com escala e deslocamento aprendidos). Isso mantém os números numa faixa estável
ao longo das camadas e acelera o treino (training).

### Bloco, pilha e saída

Um **bloco** é atenção + MLP, cada um com LayerNorm e residual. Empilhamos 4. No
fim, uma LayerNorm e uma camada linear transformam o vetor de cada posição nos 144
*logits* do próximo caractere, como no bigrama.

### Inicialização

Os pesos começam pequenos (desvio 0,02). Assim, no início, todas as saídas são
quase iguais e o modelo começa equivalente ao **uniforme**: perda (loss) ≈ ln 144 = 4,97.
Um teste confere isso, e o primeiro passo do treino real deu 5,02.

---

## 3. Testes que provam que o modelo está montado certo

[`tests/test_gpt.py`](../../tests/test_gpt.py):

| Teste | O que garante |
|---|---|
| causalidade | mudar um caractere futuro não altera as previsões anteriores |
| pesos de atenção | cada linha soma 1 e é zero acima da diagonal |
| perda inicial | ≈ ln V, o mesmo que o modelo uniforme |
| memorização | o modelo decora um lote (batch) pequeno (perda < 0,1); se não decorasse, haveria um erro no gradiente |
| geração | respeita o contexto máximo e é reproduzível com a mesma semente (seed) |

Um desses testes pegou um erro real: os alvos `ids[:, 1:]` são uma fatia, não
contígua na memória, e `view` falhava; o treino real teria quebrado do mesmo jeito.

---

## 4. O loop de treino

Código: [`src/machado_gpt/train_char.py`](../../src/machado_gpt/train_char.py).

| Elemento | O que faz |
|---|---|
| lote | B = 32 janelas (windows) de 129 caracteres sorteadas do treino: entrada = 128 primeiros, alvo = os mesmos deslocados uma posição; cada janela dá 128 exemplos |
| AdamW | descida de gradiente (gradient descent) que guarda, para cada parâmetro, uma média do gradiente e do seu tamanho, e adapta o passo de cada um; o *weight decay* encolhe levemente as matrizes de peso |
| taxa de aprendizado (learning rate) | sobe em 100 passos (*warmup*), depois desce em cosseno de 10⁻³ a 10⁻⁴ |
| corte do gradiente (gradient clipping) | se a norma passar de 1, o gradiente é reduzido |
| *dropout* 0,1 | durante o treino, zera 10% das ativações ao acaso, para o modelo não depender de detalhes do treino |
| avaliação | a cada 250 passos, estimativa em 20 lotes (batches) de treino e de validação; no fim, a perda exata em **toda** a validação, em janelas sem sobreposição |

O passo de atualização usa o AdamW pronto do PyTorch (é só a regra de atualização);
o loop (lote, perda, gradiente, corte, passo, avaliação) é escrito à mão.

### O tamanho do modelo foi escolhido pela máquina

Medimos o tempo por passo nesta CPU (Intel Core Ultra 7 155H, notebook, no WSL):

| Modelo | Parâmetros (parameters) | ms por passo | 5.000 passos |
|---|---:|---:|---:|
| 4 camadas, C = 128, T = 128, B = 64 | 0,85 M | 977 | 81 min |
| 6 camadas, C = 192, T = 256, B = 32 | 2,8 M | 3.454 | 4,8 h |
| 6 camadas, C = 384, T = 256, B = 32 (o da aula do Karpathy) | 10,9 M | 7.731 | 10,7 h |

E o número de threads:

| Threads | 1 | 4 | 8 | 11 | 22 |
|---|---:|---:|---:|---:|---:|
| ms por passo | 1.630 | 1.054 | 1.029 | **892** | 2.608 |

O processador tem 11 núcleos físicos com 2 threads cada, misturando núcleos de
desempenho e de eficiência; usar as 22 threads é **mais lento** que 11, porque elas
disputam memória. Escolhemos o modelo de 0,85 M parâmetros, com lote de 32 (~0,6 s
por passo) e 3.000 passos (~30 min). É pequeno para um GPT, e é por isso que o
treino principal (etapa 5) vai para a GPU do Colab.

---

## 5. Resultado do treino

```bash
uv run python -m machado_gpt.train_char   # ~25 min nesta CPU
```

| Passo | Treino | Validação | Minutos |
|---:|---:|---:|---:|
| 0 | 5,016 | 5,016 | 0 |
| 250 | 2,353 | 2,366 | 2 |
| 500 | 2,123 | 2,141 | 4 |
| 1.000 | 1,863 | 1,871 | 7 |
| 1.500 | 1,702 | 1,740 | 13 |
| 2.000 | 1,622 | 1,643 | 17 |
| 2.500 | 1,573 | 1,600 | 21 |
| 3.000 | 1,555 | 1,579 | 25 |

(Estimativas em 20 lotes; por isso a curva oscila um pouco, como entre os passos
2.500 e 2.750.)

- **No passo 250 o GPT já empata com o bigrama** (2,366 contra 2,383): aprender pares
  de letras é a primeira coisa que ele faz.
- **Treino e validação andam juntos** (diferença de ~0,02 a 0,03): com 0,85 M
  parâmetros e 7,4 milhões de caracteres, não há *overfitting*. O modelo é que é
  pequeno, e a curva ainda desce: mais passos ou um modelo maior dariam mais.

### Comparação com a linha de base (validação inteira)

| Modelo | Parâmetros | nats/caractere | bits/byte | perplexidade (perplexity) |
|---|---:|---:|---:|---:|
| uniforme | 0 | 4,970 | 6,95 | 144,0 |
| bigrama por contagem | 20.736 | 2,383 | 3,33 | 10,8 |
| **GPT por caracteres** | **846.592** | **1,588** | **2,22** | **4,9** |

O GPT fica **0,795 nats abaixo do bigrama** (−33%). Em perplexidade, a hesitação cai
de ~11 para ~5 caracteres. Critério da etapa atendido.

### Amostras lado a lado (semente 1337, começando em "Capitu")

**Bigrama:**

> Capituempo Gomené ssodintiramivim, quao ncitomintre impartumatititos fo J. O
> qumamorgoraçõe dera, hor de arado

**GPT:**

> Capitu.
>
> — Moço!
>
> D. Ginária?
>
> — A quando chamanha ainda?
>
> — Podia o fora. O que é prorfim; e que hor de supera de salveu de hora?

O que mudou:

- **Palavras:** a maior parte é português de verdade ("moço", "ainda", "podia",
  "esperadas", "próprias", "amigo"); as inventadas soam portuguesas ("chamanha",
  "sentimório").
- **Estrutura:** diálogos com travessão, cada fala num parágrafo, pontuação de
  pergunta e exclamação, tratamento "D." antes de nome próprio. Isso exige lembrar
  dezenas de caracteres para trás, o que o bigrama não consegue.
- **Sem sentido:** as frases não têm coerência. Com 0,85 M parâmetros e contexto de
  128 caracteres, o modelo aprendeu a forma, não o conteúdo.

---

## 6. O que as cabeças de atenção aprenderam

```bash
uv run python -m machado_gpt.inspect_attention
```

Para 64 trechos da validação, medimos onde cada cabeça põe o seu peso
([`src/machado_gpt/inspect_attention.py`](../../src/machado_gpt/inspect_attention.py)):

| Camada.cabeça | Distância média | O que faz |
|---|---:|---|
| 0.3 | 2,7 | **97% no caractere anterior** |
| 0.1 | 2,7 | 84% no caractere anterior |
| 0.2 | 3,4 | **96% exatamente dois caracteres atrás** |
| 0.0 | 5,0 | espalha entre 1 e 4 caracteres atrás |
| 1.3 | 6,9 | **40% no espaço mais recente**, ou seja, no começo da palavra atual |
| demais da camada 1 | 8 a 12 | contexto médio |
| camada 2 | 10 a 12 | contexto médio |
| camada 3 | 15 a 18 | o contexto mais longo |

Ninguém programou esses papéis; eles surgiram do treino:

- **A primeira camada monta uma janela local precisa:** uma cabeça copia o caractere
  anterior, outra o de dois atrás, outra junta os últimos quatro. É uma versão
  aprendida de "olhar as últimas letras", que o bigrama fazia com uma só.
- **A segunda camada encontra o começo da palavra:** saber onde a palavra começou
  ajuda a prever como ela termina.
- **Quanto mais funda a camada, mais longe ela olha:** a distância média cresce de ~3
  (camada 0) para ~16 (camada 3). As camadas de cima combinam informação já
  processada pelas de baixo.

**Lição:** a atenção não é uma caixa-preta inteira. Com uma medida simples (para onde
vai o peso), dá para ver funções concretas em cada cabeça.

---

## 7. Arquivos e reprodução

- [`results/03-gpt-char.json`](../../results/03-gpt-char.json) (versionado):
  configuração, curva de perda, perda na validação inteira, amostras (samples) e SHA-256 do
  checkpoint.
- `runs/03-gpt-char/` (fora do Git): checkpoint `model.pt` e log do treino. O
  checkpoint é carregado com `torch.load(..., weights_only=True)`, que não executa
  código.

```bash
uv sync
uv run pytest                              # 104 testes
uv run python -m machado_gpt.train_char    # treino (~25 min)
uv run python -m machado_gpt.inspect_attention
```

## Etapa 3: critérios de pronto (PRD)

| Critério | Situação |
|---|---|
| Atenção, blocos, normalização e resíduos implementados e explicados | seções 2 e 3 |
| Loss de validação abaixo do bigrama | 1,588 contra 2,383 nats/caractere (seção 5) |
| Amostras comparadas | seção 5 |

## Próximo passo

**Etapa 4: tokenizador (tokenizer) BPE.** Até aqui, o símbolo é o caractere: o modelo gasta
capacidade aprendendo a soletrar. O BPE junta sequências frequentes de bytes em
símbolos únicos ("que", "ção", " de"), o que deixa cada posição do contexto
carregar mais texto. O tokenizador será treinado só no conjunto de treino, e a
comparação com esta etapa será em bits por byte.
