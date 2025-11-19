# Classical Logo Detector (OpenCV + SVM)

## Propósito del proyecto
El objetivo es construir un sistema clásico (sin redes neuronales) capaz de detectar y clasificar logotipos en imágenes. Para ello se combinan detectores tradicionales de OpenCV (MSER, contornos y agrupaciones de keypoints) con descriptores BoW+HSV+HOG y un clasificador SVM multiclase. Un filtro binario adicional (SVM lineal calibrado) decide si un candidato es logo o fondo antes de pasar al clasificador final.

## Dataset disponible
El repositorio asume una estructura tipo Pascal/VOC en `data/`:

```
data/
├── train/
│   ├── brand_1/
│   │   ├── xxx.jpg
│   │   └── xxx.xml
│   └── ...
├── valid/
└── test/
```

Cada imagen tiene su XML con anotaciones `object/name` y caja `xmin,ymin,xmax,ymax`. El dataset cubre ~30 clases de logos automoción/retail (Ferrari, Milka, FedEx, etc.) y está equilibrado en tres splits (`train`, `valid`, `test`). Durante la fase de preparación generamos:
- `data/interim/annotations.csv`: CSV con todas las anotaciones fusionadas.
- `data/processed/patches_manifest.csv` + `data/processed/patches/`: parches 128×128 por logo, usados para entrenar el SVM multiclase.

## Estructura del repositorio
```
.
├── data/                # dataset VOC original + artefactos interim/processed
├── logo_detector/       # paquete Python con toda la lógica (paths, data prep, features, detector)
│   ├── data_prep.py     # parsea VOC, recorta parches
│   ├── features.py      # BoW+HSV+HOG y utilidades
│   ├── classifier.py    # entrenamiento del SVM multiclase (OpenCV)
│   └── detector.py      # pipelines MSER/contornos/keypoints + filtros y evaluación
├── models/              # artefactos entrenados (vocab, SVM, filtro binario, stats)
├── reports/figures/     # visualizaciones generadas (detecciones vs GT, etc.)
├── requirements.txt     # dependencias (opencv-python, numpy, pandas, scikit-learn, matplotlib)
└── main.py              # CLI central para reproducir todo el flujo
```

## Flujo completo (comandos reproducibles)

1. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preparar anotaciones (fusiona XMLs en un CSV)**
   ```bash
   python3 main.py prepare-annotations
   ```
   Genera `data/interim/annotations.csv` con todas las cajas y splits.

3. **Recortar parches de logos**
   ```bash
   python3 main.py crop-patches
   ```
   Crea `data/processed/patches_manifest.csv` y los recortes 128×128 en `data/processed/patches/`.

4. **Entrenar el clasificador BoW+HSV+HOG + SVM (multiclase)**
   ```bash
   python3 main.py train-classifier
   ```
   Guarda vocabulario, medias/sigmas, orden de clases y el SVM en `models/`, además de prototipos de color por clase (`models/color_prototypes.json`) usados en el re-ranking.

5. **Entrenar el detector clásico (MSER + contornos + keypoints + filtro binario)**
   ```bash
   python3 main.py train-detector
   ```
   Produce `models/logo_filter.joblib` (SVM lineal calibrado) y recopila métricas del filtro.

6. **Evaluar el pipeline completo**
   ```bash
   python3 main.py evaluate --split test \
     --bin-thresh 0.85 --min-kp 8 --topk 1 --global-nms 0.5
   ```
   Ajusta los parámetros (`--bin-thresh`, `--global-nms`, etc.) según el balance precisión/recall que busques.

7. **Inspeccionar detecciones en imágenes aleatorias**
   ```bash
   python3 main.py detect 5 \
     --bin-thresh 0.9 --min-kp 9 --global-nms 0.35 \
     --output-dir reports/figures/demo --show
   ```
   Muestra lado a lado la Ground Truth (verde) y las predicciones (rojo); guarda las figuras en `reports/figures/demo`. Quita `--show` si sólo quieres los archivos. Para un único archivo, usa `python3 main.py detect 0 --image ruta.jpg [flags]`.

8. **Modo V3 – clasificación con cajas conocidas (oracle)**
   Si quieres medir sólo la parte de clasificación (suponiendo detecciones perfectas), usa:
   ```bash
   python3 main.py oracle-classify --split test --save-errors reports/misclasificaciones.csv
   ```
   El comando recorta cada ground-truth, lo clasifica con el SVM y muestra la accuracy total y por clase. Es útil como upper bound y para depurar clases difíciles.

   Para visualizar esas predicciones sobre las propias imágenes:
   ```bash
    python3 main.py oracle-visualize --split test --limit 10 --show \
      --output-dir reports/figures/oracle
   ```
   GT y predicción (correcta en verde, incorrecta en rojo) se dibujan directamente sin pasar por el detector.

## Versiones del pipeline

| Versión | Cambios principales | Métricas orientativas (split test) |
| --- | --- | --- |
| **V1** | Propuestas MSER + contornos + keypoints; SVM BoW+HSV+HOG; filtro binario clásico | Config. balanceada (por defecto): P≈0.23 / R≈0.18 / F1≈0.20. Config. alta precisión (`--bin-thresh 0.9`, `--global-nms 0.35`, `--min-kp 9`): P≈0.30 / R≈0.10. |
| **V2** | Añade generador de texto (gradientes y morfología), sliding windows con puntuación por gradiente y re-ranking por prototipos de color (se descartan predicciones cuya paleta no coincide con la clase). El filtro binario reduce el umbral en cajas dominantes (>50 % del frame) para cubrir carteles completos y existe un NMS cruzado para eliminar duplicados dentro del mismo letrero. | Config. balanceada (por defecto): P≈0.20 / R≈0.13 / F1≈0.16 (más recall en logos pequeños y carteles). Config. alta precisión (`--bin-thresh 0.9`, `--global-nms 0.35`, `--min-kp 9`): P≈0.20 / R≈0.08 / F1≈0.11. Config. recall alto (`--bin-thresh 0.761`, `--topk 2`, `--global-nms 0`, `--no-keyprops`): P≈0.13 / R≈0.25 / F1≈0.17. |
| **V3 (oracle)** | Reutiliza el mismo SVM y descriptores, pero clasifica sólo con las cajas ground-truth para medir el techo del clasificador (sin etapa de detección). Se lanza con `python3 main.py oracle-classify --split test`. | Accuracy ≈ 0.95 en los primeros 20 ejemplos del split test (sirve como upper bound del sistema). |

> Nota: V2 prioriza cubrir casos difíciles (carteles de texto completos y logos muy grandes). Si necesitas el comportamiento más estricto de V1, basta con desactivar los nuevos generadores (`--global-nms 0.5 --bin-thresh 0.85 --min-kp 8 --topk 1`).

## Notas finales
- Todos los artefactos se regeneran con los pasos anteriores; no hay notebooks en el flujo actual.
- Puedes desactivar cada tipo de candidato con los flags `--no-keyprops`, `--no-textprops`, `--no-slideprops` en `detect/evaluate` para comparar su impacto.
- Ajusta los flags de `detect/evaluate` para adaptarte a precisión vs recall. La carpeta `reports/figures/` contiene los ejemplos generados (GT vs pred).
- Para ver ejemplos de comandos y diferencias entre versiones (V1, V2, V3), consulta `README_variantes.md`.
