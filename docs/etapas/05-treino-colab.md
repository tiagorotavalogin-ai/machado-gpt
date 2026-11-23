# Etapa 5: Treino e avaliação no Colab

Diário técnico da etapa 5. Cada seção explica o conceito, depois o que foi feito, o
resultado medido e a lição. As decisões formais estão em
[DECISOES.md](../DECISOES.md).

**Objetivo da etapa:** treinar o GPT com o tokenizador (tokenizer) BPE da etapa 4 numa
GPU, com checkpoints verificáveis por SHA-256, e comparar com o GPT por caracteres da
etapa 3 em **bits por byte**.

---

## 1. Por que uma GPU, e o que muda com ela

O treino de um Transformer é quase todo multiplicação de matrizes. A CPU faz poucas contas
por vez; a GPU faz milhares em paralelo.

Na etapa 3, o limite era a **computação (compute)**: o modelo de 0,85 M de parâmetros
(parameters) treinou 25 minutos na CPU e terminou com treino (1,55) e validação
(validation, 1,58) quase iguais. Ele não teve tempo de decorar nada.

Na GPU, o limite se inverte. Uma conta aproximada (custo ≈ 6 × parâmetros × símbolos
(tokens) processados) mostra que um modelo de 12 M vendo o treino ~15 vezes cabe em
minutos numa T4. O limite passa a ser a **quantidade de dados**: só 2,25 milhões de
símbolos de treino. Essa é a lição central desta etapa.

Uma medida local confirma o custo: 2 passos do modelo pequeno na CPU levaram ~4,7
minutos, quase todos gastos nas duas avaliações sobre a validação inteira (382 mil
símbolos). Na GPU, cada avaliação leva segundos.

---

## 2. Que tamanho de modelo?

### As duas forças

- **Capacidade:** mais parâmetros representam padrões mais sutis.
- **Dados:** o estudo Chinchilla (2022) concluiu que um treino eficiente usa ~20 símbolos
  por parâmetro. Para 2,25 M de símbolos, isso daria um modelo de ~110 mil parâmetros.

Com poucos dados, o modelo vê o mesmo texto várias vezes (epochs) e, em algum momento,
começa a **decorar** o treino em vez de aprender padrões gerais: é o sobreajuste
(overfitting). A perda (loss) de treino continua caindo, e a de validação para de cair ou
sobe. Quanto maior o modelo, mais cedo isso acontece.

### As opções

Cada camada (layer) tem ~12·C² parâmetros (4·C² na atenção (attention), 8·C² no MLP).

| | camadas | C | contexto (context) | parâmetros | símbolos por parâmetro |
|---|---:|---:|---:|---:|---:|
| pequeno | 4 | 256 | 256 | 4,27 M | 0,5 |
| médio | 6 | 384 | 256 | ~12 M | 0,19 |
| grande | 8 | 512 | 512 | ~27 M | 0,08 |

O contexto de 256 símbolos cobre ~860 bytes, uns dois parágrafos.

### A escolha: treinar o pequeno e o médio, e comparar

Na prática, com dados escassos, um modelo um pouco grande demais e bem regularizado
costuma ganhar de um pequeno: o excesso é controlável, a falta de capacidade não. Mas em
vez de supor, medimos, como na escolha do vocabulário (vocabulary).

O grande fica de fora por três motivos:

- **sobreajuste:** decoraria ainda mais cedo;
- **custo:** o contexto de 512 dobra o custo da atenção, que cresce com o quadrado do
  contexto, e só ajuda se o modelo usar contexto longo. Na etapa 3, a última camada
  olhava em média ~16 caracteres para trás;
- **limites do Colab gratuito:** sessões que caem e cota diária de GPU.

Ele entra se o médio terminar sem sinal de sobreajuste.

---

## 3. O que muda no modelo

O modelo é o mesmo [`gpt.py`](../../src/machado_gpt/gpt.py) da etapa 3, com duas mudanças:

- **Dropout 0,2** (era 0,1): mais regularização, porque agora o risco é decorar.
- **Matriz compartilhada (weight tying):** a camada de saída usa a mesma matriz da
  embedding de símbolos. O vetor que representa "Capitu" na entrada é o mesmo que pontua
  "Capitu" na saída. Isso economiza V × C parâmetros (1,05 M no pequeno) e, com poucos
  dados, ajuda os símbolos raros: os dois usos treinam o mesmo vetor. O GPT-2 faz o
  mesmo.

Um detalhe de memória: a etapa 3 guardava os pesos da atenção (attention weights) de toda
chamada, para a inspeção das cabeças (heads). Agora isso só acontece fora do treino; no
treino, com lotes (batches) de 64 × 256, seriam centenas de MB de GPU desperdiçados.

---

## 4. O que muda no treino

Código: [`src/machado_gpt/train_bpe.py`](../../src/machado_gpt/train_bpe.py). O laço
(loop) é o da etapa 3: sortear janelas (windows), prever, calcular a perda, gradiente,
AdamW. As adições:

### Precisão mista (mixed precision)

Na GPU, as contas pesadas são feitas em **float16** (16 bits) em vez de float32: metade da
memória e cerca do dobro da velocidade numa T4. O PyTorch (`autocast`) mantém em float32
as operações sensíveis, como o softmax e a perda.

O risco: o float16 não representa números muito pequenos, e gradientes minúsculos viram
zero. O **gradient scaler** multiplica a perda por um fator grande antes do backward (os
gradientes crescem junto), e divide de volta antes do corte de gradiente e do passo do
otimizador (optimizer). Se aparecer um infinito, ele pula aquele passo e reduz o fator.

### Avaliação exata

A cada 100 passos, a perda é calculada sobre a **validação inteira** (382 mil símbolos),
não sobre uma amostra. A curva fica sem ruído de amostragem, e a escolha do melhor ponto
fica justa. A perda de treino continua estimada por 24 lotes sorteados.

### Parada antecipada (early stopping)

Se a validação não melhora por 10 avaliações seguidas (1.000 passos), o treino para. O
`best.pt` guarda sempre o melhor ponto pela validação, que é o modelo que vale.

### Retomada (resume)

A sessão gratuita do Colab cai por inatividade ou depois de algumas horas. O `last.pt`,
salvo a cada 500 passos, guarda tudo o que é preciso para continuar como se nada tivesse
acontecido:

- pesos (weights), estado do otimizador (os momentos do AdamW) e do gradient scaler;
- os **estados aleatórios**: o gerador que sorteia as janelas e o do PyTorch, que decide o
  dropout. Sem eles, o treino retomado seria outro.

Cada checkpoint é escrito num arquivo temporário e só depois renomeado, para que uma queda
no meio da gravação nunca deixe um arquivo pela metade.

**Teste:** [`tests/test_train_bpe.py`](../../tests/test_train_bpe.py) simula uma queda no
passo ~15, retoma do `last.pt` do passo 10 e verifica que os pesos finais são **idênticos**
aos de um treino que nunca parou.

---

## 5. Os dados no Colab: regenerar em vez de copiar

O corpus e os arquivos codificados nunca entram no Git nem no Drive. O notebook refaz o
caminho das etapas 1 e 4 no Colab (download com hash, extração, divisão (split),
tokenização) e confere o resultado:

1. o catálogo `data/catalog/works.csv`, versionado, tem o SHA-256 do texto limpo de cada
   obra. Depois da extração no Colab, `git diff` do catálogo deve sair vazio;
2. os SHA-256 do tokenizador e dos três conjuntos devem bater com `results/04-bpe.json`;
3. o script de treino confere esses hashes de novo e se recusa a treinar se algo diferir
   (com teste).

Para que a extração seja idêntica, o notebook instala as mesmas versões do `pdfplumber` e
do `pdfminer.six` do `uv.lock`. Assim, a etapa 5 também testa a **reprodutibilidade** do
projeto inteiro numa máquina nova.

### Acesso ao repositório privado

O Colab precisa de uma credencial para clonar um repositório privado. Usamos um token
*fine-grained* do GitHub, só de leitura e só deste repositório, guardado nos Segredos do
Colab. Ele vai num cabeçalho (header) do `git clone` e não fica gravado no `.git/config`
nem no notebook.

---

## 6. A primeira medida: o modelo antes de treinar

Antes do primeiro passo, a perda do modelo pequeno foi **8,40 nats por símbolo**. Um
modelo que não sabe nada dá a mesma chance a cada um dos 4.096 símbolos, e a perda seria
ln 4.096 = 8,32. A inicialização com pesos pequenos chega perto disso, como esperado.

Em bits por byte, o modelo sem treino dá 3,60, pior que o bigrama (3,33). Um modelo grande
não começa sabendo nada.

---

## 7. Resultados

_A preencher depois do treino no Colab._

---

## 8. Arquivos e reprodução

- [`notebooks/05-colab.ipynb`](../../notebooks/05-colab.ipynb): o roteiro no Colab, com o
  porquê de cada passo. Versionado sem saídas (outputs).
- [`src/machado_gpt/train_bpe.py`](../../src/machado_gpt/train_bpe.py): o treino.
- `results/05-gpt-bpe-small.json` e `results/05-gpt-bpe-medium.json` (versionados):
  configuração, curvas, métricas de validação e teste, amostras (samples), commit do
  código, SHA-256 dos dados e do `best.pt`.
- Checkpoints: no Drive do proprietário, fora do Git.

---

## Etapa 5: critérios de pronto (PRD)

| Critério | Situação |
|---|---|
| Treino no Colab com checkpoints e hash SHA-256 | código e notebook prontos; treino pendente |
| Métricas por byte para comparar com a etapa 3 | código pronto (bits por byte em validação e teste) |
| Relatório de resultados com amostras | pendente |
