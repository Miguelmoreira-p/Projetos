# Treinamento de modelo customizado — EduPhone Guard

Esta pasta contém a estrutura para treinar um detector de celulares
especializado no domínio escolar, para substituir o modelo genérico
pré-treinado no COCO usado inicialmente.

## Por que treinar um modelo próprio

O modelo genérico (`yolov8n.pt`, treinado no COCO) já reconhece a classe
"cell phone", mas **não foi validado para ambiente escolar real**: ângulos
de câmera de sala de aula, distâncias, iluminação e oclusão parcial (celular
na mão, meio escondido, sobre a mesa) diferem bastante do dataset genérico.
Não assuma que o modelo genérico terá precisão adequada em produção — trate-o
como ponto de partida, não como solução final.

## Fontes de dados recomendadas (nunca imagens reais de estudantes)

- **COCO** (classe `cell phone`) — CC BY 4.0.
- **Open Images V7** (classe `Mobile phone`) — CC BY 4.0.
- **Imagens produzidas em ambiente controlado**: voluntários adultos, com
  consentimento explícito, cobrindo diferentes modelos de celular, ângulos,
  distâncias, iluminações e níveis de oclusão.

## Fluxo

```bash
# 1. Organizar dados brutos (imagens + labels YOLO) em um dataset válido
python model/training/prepare_dataset.py \
    --raw-images /caminho/para/imagens \
    --raw-labels /caminho/para/labels \
    --output model/datasets/school_phones

# 2. Treinar (transfer learning a partir do YOLOv8 pré-treinado)
python model/training/train.py \
    --data model/datasets/school_phones/data.yaml \
    --epochs 100 \
    --output-name custom_school_phone_v1

# 3. Avaliar — preste atenção especial à False Positive Rate
python model/training/evaluate.py \
    --weights model/custom_school_phone_v1.pt \
    --data model/datasets/school_phones/data.yaml

# 4. Exportar para produção
python model/training/export.py --weights model/custom_school_phone_v1.pt --format onnx
```

## Equilíbrio entre recall e precisão

**Não otimize apenas para detectar o maior número possível de celulares.**
Um sistema com muitos falsos positivos gera fadiga de alerta na equipe
revisora e desconfiança no sistema. Ao ajustar hiperparâmetros e o limiar de
confiança (`MIN_CONFIDENCE`), avalie sempre em conjunto:

- Precision
- Recall
- mAP@0.5 e mAP@0.5:0.95
- False Positive Rate (ver `evaluate.py`)

Uma métrica adicional recomendada para produção: acompanhar, ao longo do
tempo, a proporção de eventos que revisores humanos marcam como "falso
positivo" — esse é o sinal mais confiável de qualidade real do modelo em
campo, mais do que as métricas de validação offline.

## Convenção de nomenclatura

Para que `app/vision/detector.py` selecione automaticamente
`CustomSchoolPhoneDetector` em vez do `YOLODetector` genérico, o arquivo de
pesos deve conter a palavra `custom` no nome (ex.: `custom_school_phone_v1.pt`)
e ser referenciado em `MODEL_PATH` no `.env`.
