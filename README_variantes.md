# Variantes del pipeline (V1, V2, V3)

Este documento describe las tres configuraciones principales que puedes lanzar con `main.py`. Todas comparten la misma base de features (ORB → BoW de 400 palabras + histograma HSV + HOG) y el mismo clasificador multiclase SVM (OpenCV, kernel RBF). La diferencia está en cómo generamos las propuestas o en si usamos las cajas anotadas.

## Arquitectura base

| Bloque | Descripción |
| --- | --- |
| **Extracción** | ORB (`nfeatures=1000`) → k-means (Bag of Words, K=400) + histograma HSV (8×8×8) + HOG (bloques 32×32). Cada vector se estandariza (`mu.npy`, `sigma.npy`). |
| **Clasificador** | `cv2.ml.SVM` (RBF) entrenado sobre los parches recortados (`train-classifier`). |
| **Filtro binario** | LinearSVC calibrado (scikit-learn) que decide logo vs fondo antes del SVM multiclase (`train-detector`). |
| **Generadores de candidatos** | Todo es visión clásica con OpenCV:<br>• **MSER** (`cv2.MSER_create`): detecta regiones extremas estables en intensidad, muy útil para texto/logos.<br>• **Contornos + Canny** (`cv2.Canny`, `cv2.findContours`): obtiene bordes y componentes conectados; filtramos por área y aspecto.<br>• **Agrupaciones de keypoints** (`cv2.ORB_create`): unimos zonas con muchos puntos ORB de alta respuesta.<br>• **Texto por gradientes** (`cv2.Sobel` + morfología): resalta líneas horizontales de texto y cierra huecos con `cv2.morphologyEx`.<br>• **Sliding windows**: barrido clásico en varias escalas usando mapas integrales de gradiente (`cv2.integral`). Ningún módulo usa redes neuronales. |
| **Re-ranking** | Prototipos de color por clase (`models/color_prototypes.json`) penalizan predicciones cuya paleta no coincide. |
| **Oracle** | Usa directamente las cajas GT y sólo ejecuta el SVM, sin ninguna etapa de detección. |

---

## V1 – Detector básico (MSER + Contornos + Keypoints)

- **Uso**: primer baseline, sin los nuevos módulos de texto ni sliding windows.
- **Características**:
  - Propuestas MSER + contornos + agrupaciones de keypoints.
  - Sin re-ranking por color (puedes desactivarlo con `--no-textprops --no-slideprops --no-keyprops`).
  - Ajusta `--global-nms` para controlar la cantidad de cajas.
- **Comando sugerido**:
  ```bash
  python3 main.py detect 5 \
    --bin-thresh 0.85 \
    --min-kp 8 \
    --topk 1 \
    --global-nms 0.35 \
    --no-textprops --no-slideprops --no-keyprops \
    --output-dir reports/figures/v1_demo --show
  ```
  Explicación rápida de flags:
  - `--bin-thresh`: umbral del filtro binario (sube para más precisión).
  - `--min-kp`: nº mínimo de keypoints ORB por caja.
  - `--topk`: cajas a mantener por clase tras el NMS por clase.
  - `--global-nms`: IoU para eliminar cajas entre clases diferentes.
  - `--no-*props`: desactiva módulos de propuestas concretos.

---

## V2 – Detector completo con texto, sliding windows y límite de cajas

- **Uso**: configuración recomendada para balancear recall y precisión, limitando el nº de cajas por imagen.
- **Características**:
  - Activa MSER, contornos, texto y sliding windows; puedes desactivar módulos con `--no-textprops`, `--no-slideprops`, `--no-keyprops`.
  - Re-ranking por color y reducción automática de umbral en cajas dominantes (\>50 % del frame).
  - Nuevo flag `--max-total` para limitar el nº total de cajas tras el NMS global.
- **Comando (recall alto pero ≤2 cajas)**:
  ```bash
  python3 main.py detect 5 \
    --bin-thresh 0.8 \
    --min-kp 8 \
    --topk 1 \
    --global-nms 0.2 \
    --pad 0.15 \
    --no-keyprops \
    --max-total 2 \
    --output-dir reports/figures/v2_demo --show
  ```
  - `--pad`: padding relativo aplicado a cada caja antes de extraer el parche.
  - `--max-total`: máximo nº de detecciones finales por imagen.
  - Mantén `--no-keyprops` si quieres priorizar texto + sliding y reducir ruido.

  Configuración alternativa (alta precisión):
  ```bash
  python3 main.py detect 5 \
    --bin-thresh 0.9 \
    --min-kp 9 \
    --topk 1 \
    --global-nms 0.35 \
    --max-total 2 \
    --output-dir reports/figures/v2_prec --show
  ```

---

## V3 – Oracle (clasificación únicamente)

- **Uso**: mide el techo del clasificador (sin detección). Se recorta cada GT y se pinta verde/rojo según el SVM acierte o no.
- **Comandos**:
  1. Métricas agregadas:
     ```bash
     python3 main.py oracle-classify --split test \
       --save-errors reports/misclasificaciones.csv
     ```
  2. Visualización (cada GT coloreada según predicción):
     ```bash
     python3 main.py oracle-visualize --split test \
       --limit 10 --output-dir reports/figures/oracle --show
     ```
     - `--limit`: nº de imágenes a mostrar.
     - `--show`: abre ventanas de OpenCV (ESC para salir). Sin `--show`, sólo guarda los archivos.

- **Interpretación**: este modo suele dar accuracies ≥0.9 porque elimina el error de detección. Úsalo como “upper bound” cuando documentes resultados o depures clases difíciles.

---

### Consejos generales

- Ejecuta `python3 main.py train-classifier` y `python3 main.py train-detector` después de cualquier cambio en `logo_detector/` para mantener sincronizados SVM, filtro binario y prototipos de color.
- Cambia el split (`--split train/val/test`) en los comandos de `detect`, `evaluate` u `oracle-*` para revisar distintas particiones.
- Usa `--output-dir` siempre que quieras comparar resultados entre sesiones sin mezclar archivos (el `reports/figures/` está gitignoreado).
