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
(tokens) processados) mostra que um modelo de 12 M vendo o treino dezenas de vezes cabe em
minutos numa T4 (medido: 36 voltas (epochs) em 19 minutos). O limite passa a ser a **quantidade de dados**: só 2,25 milhões de
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

**Resultado (23/11/2026):** no Colab, o catálogo saiu sem diferenças e os SHA-256 do
tokenizador e dos três conjuntos bateram. A cadeia inteira das etapas 1 a 4 (download,
extração de 110 obras, divisão e BPE) reproduziu o corpus **byte a byte** numa máquina
diferente. O único aviso foi um conflito do pip entre o Pillow 13.0.0, a mesma versão do
`uv.lock`, e o Gradio, que vem no Colab e o projeto não usa.

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

Treino em 23/11/2026, numa Tesla T4 (Colab gratuito), código no commit `bc8a709`. Os dois
modelos rodaram os 5.000 passos: 64 × 256 × 5.000 = 82 milhões de símbolos processados,
ou **36 voltas (epochs)** pelo treino.

| modelo | parâmetros | tempo | validação (bits/byte) | teste (bits/byte) |
|---|---:|---:|---:|---:|
| bigrama (etapa 2) | 21 mil | segundos | 3,33 | |
| GPT por caracteres (etapa 3, CPU) | 0,85 M | 25 min | 2,22 | |
| **GPT BPE pequeno** | 4,27 M | 8,4 min | **1,696** | 1,670 |
| **GPT BPE médio** | 12,3 M | 18,8 min | **1,663** | 1,632 |

Em nats por símbolo, na validação: 3,950 (pequeno) e 3,872 (médio); perplexidade
(perplexity) de 52 e 48 símbolos.

### Quanto do ganho é do BPE?

Do GPT por caracteres para o BPE pequeno, a queda é de 2,22 para 1,70 bits por byte (23%).
Mas três coisas mudaram juntas:

- o tokenizador;
- o modelo, 5 vezes maior;
- o treino: 82 milhões de símbolos, ~280 milhões de bytes de texto, contra 12,3 milhões
  de caracteres na etapa 3, ~20 vezes mais.

O número mede o pacote inteiro, não só o BPE. Separar os efeitos exigiria treinar o GPT
por caracteres na GPU com o mesmo orçamento; fica como experimento possível, não
necessário para o objetivo da etapa.

### As curvas: o médio aprende mais, e decora mais

Distância entre a perda de validação e a de treino (nats por símbolo):

| passo | 500 | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 |
|---|---:|---:|---:|---:|---:|---:|
| pequeno | 0,16 | 0,24 | 0,36 | 0,43 | 0,49 | 0,51 |
| médio | 0,17 | 0,27 | 0,48 | 0,65 | 0,79 | **0,85** |

- **Até o passo ~1.000, os dois são iguais.** Os dois aprendem o mesmo: os padrões
  frequentes, que servem para qualquer obra.
- **Depois, o médio desce mais no treino** (3,02 contra 3,44 no fim), mas só um pouco
  mais na validação (3,87 contra 3,95). A capacidade extra foi quase toda para **decorar**
  o treino: a distância chegou a 0,85.
- **A validação do médio praticamente parou:** melhorou 0,01 nos últimos 1.000 passos
  (o pequeno, 0,025). A parada antecipada não disparou porque ainda havia melhoras de
  milésimos, mas a curva está no platô.

**Conclusão:** triplicar os parâmetros rendeu 2% na validação (1,696 para 1,663). É o
retrato de um treino **limitado por dados**: com 2,25 milhões de símbolos, mais capacidade
quase não ajuda. Pela regra combinada, o modelo grande **não** é treinado: o médio já mostra
sobreajuste. O caminho para melhorar seria mais texto (a etapa 6, opcional, com um corpus
geral de português) e não mais parâmetros.

O teste ficou um pouco abaixo da validação nos dois modelos (1,67 e 1,63). Os conjuntos têm
obras e gêneros diferentes, e o teste parece um pouco mais previsível; o que importa é que
a ordem entre os modelos é a mesma nos dois.

### As amostras (samples)

Pequeno, a partir de "Capitu":

> Capitu estremeceu, temerta por cima, esta palavra sacudiu o pensamento deitou-a.
> Nenhuma potência, mas nenhuma maltra. [...] CAPÍTULO XVIX
>
> Rubião era de Sofia que não entendeu em afeição, mas o gesto de Sofia fazia peste. A voz
> pura que Luís Garcia vinha pela primeira parte anterior do rosto dele [...]

Médio, a partir de "Não consultes dicionários.":

> Não consultes dicionários. Entretanto, É natural que, por um simples estabelecimento
> político, são esquecer que o espírito do homem tem podido dissipar para mal a sociedade,
> e a moralidade elevadada no Cid, ou pelo menos pelo resultado. [...] A ansiedade geral
> confessa o contrário, como passa a todo reforma.

**O que os modelos aprenderam:** pontuação e ritmo de Machado, nomes de personagens,
títulos de capítulo, vocabulário do autor ("imprudência", "desvanecimentos", "sagacidade
literária") e o tom de cada gênero: o segundo trecho do médio soa como crítica ou crônica.

**O que falta:**

- **coerência:** o pequeno junta Rubião e Sofia (*Quincas Borba*) com Luís Garcia (*Iaiá
  Garcia*) na mesma cena; o médio erra menos (Luís Garcia aparece com Iaiá, da mesma obra);
- **palavras inventadas:** "bonitosas", "divórdia", "elevadada";
- **algarismos romanos impossíveis:** "XVIX".

É o limite previsto no PRD: estilo local, sem sentido global nem conhecimento de mundo.

As amostras de um mesmo modelo repetem trechos ("Nenhuma potência, mas nenhuma maltra")
porque usam a mesma semente (seed): depois que o começo diferente sai do foco, os mesmos
números sorteados levam às mesmas escolhas.

### Checkpoints

| modelo | SHA-256 do `best.pt` |
|---|---|
| pequeno | `72655aafcb911aa6f5d637b5d71aa4fd63c28cdce7421bce0050ccf042e086cb` |
| médio | `7c882e88c44be69ec7881c1252f733193d7de47b470d1ff5feaf42d8ad93d54d` |

Os arquivos ficam no Drive do proprietário. Qualquer cópia pode ser conferida contra esses
hashes antes de ser carregada.

### Lições

1. **Com GPU, o gargalo vira os dados.** Na CPU, faltava tempo; na T4, 36 voltas pelo
   treino em 19 minutos, e o limite passou a ser o tamanho da obra de Machado.
2. **Mais parâmetros com poucos dados vira memória, não generalização.** O médio decorou
   mais (distância de 0,85) e generalizou só 2% melhor.
3. **Medir a distância entre treino e validação, não só a validação.** A validação sozinha
   diria "o médio é melhor"; a distância mostra a que custo.

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
| Treino no Colab com checkpoints e hash SHA-256 | seção 7: dois modelos na T4, `best.pt` com SHA-256 |
| Métricas por byte para comparar com a etapa 3 | seção 7: 2,22 (caracteres) para 1,663 bits/byte (médio) |
| Relatório de resultados com amostras | seção 7 |

## Próximo passo

A etapa 6 é opcional: pré-treino num corpus geral de português, seguido de ajuste
(fine-tuning) com Machado. Os resultados desta etapa dizem que é o caminho com mais
potencial, porque o limite agora é a quantidade de texto.
