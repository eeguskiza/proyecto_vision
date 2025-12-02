# Documentación técnica (logo_detector)

Este paquete reúne todo el flujo clásico de un detector de logotipos: preparación de datos, extracción de descriptores, entrenamiento de SVMs y ejecución del pipeline de detección + clasificación. Todo se apoya en OpenCV (MSER, ORB, HOG, contornos) y dos SVMs (uno binario logo/fondo y otro multiclase). El objetivo es poder medir por separado detección y clasificación, y combinarlas en un pipeline reproducible.

## Visión general del flujo
1. Fusionar XMLs VOC en un CSV único y recortar parches por logo para entrenar el clasificador.
2. Aprender un vocabulario BoW (ORB) y combinarlo con HSV + HOG para generar descriptores robustos.
3. Generar propuestas de detección (MSER/contornos/gradientes), filtrarlas con un SVM binario logo/fondo y etiquetarlas con el SVM multiclase.
4. Aplicar NMS (por clase y global) y re-ranking ligero por color para depurar duplicados y penalizar colores incoherentes.
5. Evaluar por separado: modo oracle (clasificación con GT) y modo pipeline (detección + clasificación).

## Detector (detector.py)
- **Propuestas**: MSER, contornos y módulos opcionales (keypoints, texto, sliding) generan cajas candidatas en varias escalas.
- **Filtro binario**: un LinearSVC calibrado (scikit-learn) decide logo vs fondo antes de pasar al multiclase; umbrales y mínimos de keypoints controlan la precisión.
- **Postproceso**: NMS por clase y global, límite de cajas totales y re-ranking por prototipos de color para penalizar paletas inconsistentes.
- **Presets**: `mser_preset` controla la agresividad de MSER (`strict` para precisión, `balanced` para más recall).

## Clasificador (classifier.py + features.py)
- **Descriptores**: BoW de ORB (controlado por `vocab_size`, `orb_features`), histograma HSV y HOG opcional para textura.
- **Entrenamiento**: SVM RBF multiclase; se guardan vocabulario (`bow_dict.yml`), medias/sigmas (`mu.npy`, `sigma.npy`), modelo (`svm_bow_hsv_hellinger.xml`) y prototipos de color por clase.
- **Oracle**: `oracle-classify` y `oracle-visualize` evalúan el techo del clasificador sin depender de la etapa de detección.

## Datos y rutas
- `data_prep.py`: fusiona anotaciones y recorta parches; escribe en `data/interim` y `data/processed`.
- `paths.py`: rutas centralizadas a datos, modelos y artefactos (`models/`), y helper para crear la estructura necesaria.

## Ejecutar todo en un paso
```bash
python train_test.py
```
Ese comando prepara anotaciones, recorta parches, entrena clasificador y detector, y evalúa el pipeline.

## Flujo recomendado paso a paso
1. `python main.py prepare-annotations` y `python main.py crop-patches` para dejar los datos listos.
2. `python main.py train-classifier` para generar vocabulario, SVM multiclase y prototipos de color.
3. `python main.py train-detector` para entrenar el filtro binario logo/fondo.
4. `python main.py detect 5 --show` para una inspección rápida o `python main.py evaluate --split test` para métricas P/R/F1.
