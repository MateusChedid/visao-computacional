# RPG Dice Vision

Detecção de resultados de dados de RPG (d6, d8, d10, d12, d20) em tempo real via webcam, usando YOLOv8.

> **d10**: o valor `0` representa **zero** (não dez).

---

## Visão geral da abordagem

O projeto evoluiu por várias iterações. Os pontos-chave da arquitetura atual:

- **Uma única ROI (octógono)**: definida uma vez em `utils/roi_inference.py`, usada tanto na coleta quanto na inferência — garante que o dado apareça na mesma escala em treino e uso real.
- **Bordas com cor real do papel**: a área fora do octógono é preenchida esticando os pixels reais da borda (`cv2.BORDER_REPLICATE`), evitando bordas artificiais que confundem o detector.
- **5 posições de captura por face**: centro + 4 cantos diagonais (NO, NE, SO, SE) — cobre variações de perspectiva e posicionamento do dado dentro do tray.
- **90 rotações sintéticas por foto** (passo 4°), geradas a partir de um quadrado central "seguro" (`lado/√2`) — evita pixels replicados/distorcidos nas bordas após rotação.
- **Pré-processamento em escala de cinza**: todas as imagens do dataset são convertidas para P&B (3 canais BGR iguais) antes do treino, e o frame da webcam passa pela mesma conversão antes de ser enviado ao modelo — foca o aprendizado em forma/contraste/textura dos números, ignorando cor.
- **Detecção automática de bbox por rotação**: 3 níveis de confiança (0.15 → 0.05 → 0.001), `imgsz=960`, filtrada por região calibrada por tipo de dado **e** por posição (centro/cantos). A região de busca é projetada geometricamente para cada ângulo de rotação via `rotate_rect()`.
- **Download automático do dataset**: se o dataset não estiver presente localmente, `train.py` baixa automaticamente o ZIP do Google Drive (configurado em `config.yaml`).

---

## Estrutura

```
rpg_dice_cv_v3/
├── dataset_collector/
│   ├── auto_collect.py          ← coleta: 5 fotos/face → 90 rotações + bbox automática
│   ├── augment_offline.py       ← gera variações de brilho/contraste/ruído
│   ├── check_small_bboxes.py    ← lista imagens com bbox menor que a média
│   ├── convert_to_grayscale.py  ← converte dataset para escala de cinza
│   ├── remove_augmented.py      ← remove variações offline do dataset
│   ├── validate_dataset.py
│   └── input/                   ← fotos originais (d6_1.jpg, d6_1_a.jpg, …)
├── dataset/
│   ├── dataset.yaml             ← 57 classes
│   └── images/ labels/          ← train / val / test
├── training/
│   ├── train.py                 ← treino com download automático do Drive
│   └── config.yaml
├── inference/
│   ├── detect.py                ← inferência em tempo real (P&B interno, cores na tela)
│   └── result_reader.py
├── utils/
│   └──roi_inference.py         ← octógono (única ROI: coleta + inferência)
└── requirements.txt
```

---

## Modelo Treinado (Checkpoints)

O arquivo de pesos da rede neural (`best.pt`) não está incluído no repositório
por ser grande (~87MB). Faça o download pelo link abaixo e coloque na pasta raiz
do projeto antes de rodar a inferência:

**Download:** https://drive.google.com/file/d/1dXmP8hL1PpqbzRj5RkV4thX5ydOh38b_/view?usp=sharing

| Arquivo | Descrição | Época |
|---------|-----------|-------|
| `best.pt` | Melhor modelo (usado na inferência) | 25/67 |

### Como usar após o download

Coloque o arquivo na raiz do projeto e execute:

```bash
python utils/roi_inference.py
python inference/detect.py --weights best.pt
```
---

## Dataset

O dataset de imagens de treino e validação (~25.000 imagens em escala de cinza)
não está incluído no repositório. Faça o download pelo link abaixo:

**Download:** https://drive.google.com/file/d/1Zb3Wmu4uTaL-_4BA41JewOvsW7oBkZox/view?usp=sharing

Após baixar o ZIP, extraia na pasta `dataset/` do projeto. O `training/train.py`
também pode baixar o dataset automaticamente se `drive_zip_id` estiver configurado
em `training/config.yaml`.

---

## Dados suportados

| Dado | Faces | Classes |
|------|-------|---------|
| d6 | 1–6 | 6 |
| d8 | 1–8 | 8 |
| d10 | **0**–9 | 10 |
| d12 | 1–12 | 12 |
| d20 | 1–20 | 20 |
| — | unknown | 1 |
| **Total** | | **57** |

---

## Instalação

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install gdown   # para download automático do dataset do Google Drive
```

---

## Fluxo de uso completo

### 1. Definir o octógono do tray

Na primeira vez, ou quando mudar a posição da câmera:

```bash
python utils/roi_inference.py
python utils/roi_inference.py --camera 1   # outra webcam
```

Clique nos cantos do tray em ordem, `ENTER` duas vezes para confirmar. Salvo em `camera_config.yaml`.

### 2. Capturar as fotos originais

Para cada face de cada dado (57 faces), tire **5 fotos** com o dado em posições diferentes:

```bash
python dataset_collector/auto_collect.py --camera 1
```

O programa guia cada captura com um crosshair na tela indicando onde posicionar o dado:

| Foto | Posição | Sufixo do arquivo |
|------|---------|-------------------|
| 1 | Centro | `d6_1.jpg` |
| 2 | Canto NO | `d6_1_a.jpg` |
| 3 | Canto NE | `d6_1_b.jpg` |
| 4 | Canto SO | `d6_1_c.jpg` |
| 5 | Canto SE | `d6_1_d.jpg` |

Controles durante a captura: `ESPAÇO` = fotografar · `Q` = pular posição · `ESC` = cancelar face.

Antes de processar, o programa pede calibração da **região de busca** por tipo e por posição — arraste o retângulo sobre o dado e ajuste o tamanho com scroll, `ENTER` para confirmar.

Para usar fotos já tiradas (pasta `input/` já populada):

```bash
python dataset_collector/auto_collect.py --from-folder
```

### 3. Converter para escala de cinza

```bash
python dataset_collector/convert_to_grayscale.py
```

Converte todas as imagens de `dataset/images/train/` e `val/` para P&B (sobrescreve os `.jpg`). Labels não são alterados.


### 4. Validar

```bash
python dataset_collector/validate_dataset.py
```

### 5. Treinar

```bash
python training/train.py
```

Se o dataset não estiver presente localmente, o script baixa automaticamente do Google Drive. Configure o ID do arquivo em `training/config.yaml`:

```yaml
drive_zip_id: "SEU_ID_DO_DRIVE"
```

### 8. Inferência em tempo real

```bash
python inference/detect.py                         # webcam (índice 0)
python inference/detect.py --source 1              # outra webcam
python inference/detect.py --weights caminho/best.pt
python inference/detect.py --source foto.jpg       # imagem estática
python inference/detect.py --no-roi                # ignora octógono
```

O sistema exibe o frame em **cores normais** com as bboxes sobrepostas. No **canto superior direito** aparece uma janela pequena mostrando o canvas em **preto e branco** — exatamente o que o modelo está processando internamente.

---

## Controles da inferência

| Tecla | Ação |
|-------|------|
| `ESPAÇO` | Congelar frame e mostrar resultado |
| `R` | Voltar ao live feed |
| `S` | Salvar screenshot + canvas P&B enviado ao modelo |
| `Q` | Sair |

---

## Pré-processamento P&B — como funciona

O modelo é treinado com imagens em escala de cinza. Na inferência, o fluxo é:

```
Frame da webcam (cores)
        ↓
octagon_to_square()    ← recorta o octógono, cor real do papel nas bordas
        ↓
to_grayscale_bgr()     ← converte para P&B (3 canais iguais)
        ↓
model.predict()        ← detecção
        ↓
Overlay sobre o frame original (cores) + preview P&B no canto
```

O dado é identificado apenas por forma, contraste e textura dos números — sem depender de cor do plástico.

---

## Solução de problemas

- **CUDA out of memory**: reinicie o PC/servidor. Se persistir, force CPU com `$env:CUDA_VISIBLE_DEVICES="-1"` (Windows) ou `CUDA_VISIBLE_DEVICES=-1 python ...` (Linux). Em servidores compartilhados, verifique o uso da GPU com `nvidia-smi` e ajuste `batch` em `config.yaml` conforme a memória disponível.
- **Dataset não encontrado ao treinar**: verifique se `drive_zip_id` está preenchido em `config.yaml` e se o arquivo no Drive está compartilhado publicamente. Instale `gdown` com `pip install gdown`.
- **Estrutura do dataset errada após extração** (0 imagens encontradas): rode `python debug_dataset.py` para localizar onde os arquivos foram parar, e `python fix_dataset_structure.py` para corrigir.
- **Muito fallback na coleta**: normal para dados pequenos (d10) e posições de canto. A região de busca calibrada garante que o fallback fique no lugar certo — confira os previews em `dataset_collector/_work/bbox_preview/`.
- **Detecção ruim na posição normal**: confirme que o octógono cobre o tray inteiro e que a câmera está na mesma posição usada na coleta.

---

## Declaração de Uso de IA Generativa

O arquivo `training/train.py` foi inicialmente gerado com auxílio de IA generativa
(Claude, da Anthropic), pois nenhum membro do grupo possuía experiência prévia
suficiente para estruturar de forma autônoma o pipeline de configuração e execução
do treinamento YOLOv8 via código Python.

**O que foi solicitado à IA:** organizar os parâmetros definidos em
`training/config.yaml` em um script Python que carregasse essas configurações,
verificasse a integridade do dataset antes de iniciar e executasse o treinamento
via a API da biblioteca Ultralytics.

**O que o script faz, em detalhes:**

- Lê `config.yaml` com `yaml.safe_load()` e extrai os parâmetros de treinamento
- Verifica se as pastas `dataset/images/train/` e `dataset/images/val/` existem e
  contêm imagens — se não, tenta baixar automaticamente um ZIP do Google Drive
  usando `gdown`, cujo ID é configurado via `drive_zip_id` no `config.yaml`
- Instancia o modelo YOLOv8 via `YOLO(modelo_base)` da biblioteca Ultralytics
- Chama `model.train()` passando os parâmetros lidos do YAML — épocas, batch,
  imgsz, otimizador, learning rate e augmentations (HSV, mosaico, rotação, flip)
- Salva os resultados em `runs/train/` com nome baseado na data/hora de início

Todos os membros do grupo revisaram e validaram o script gerado, entendendo o
papel de cada parâmetro — em especial as configurações de augmentation online
e sua relação com o dataset gerado offline com rotações e conversão para escala
de cinza.