# Informe de resolución y validación del Entregable 1

## Resumen ejecutivo

**Estado: listo dentro del alcance revisado.** El notebook resuelve los objetivos del enunciado, usa la base de datos y los dos scripts suministrados como referencia técnica, y se ha ejecutado de principio a fin en un kernel nuevo sin errores.

La validación final se realizó el 1 de octubre de 2026 con Python 3.12.9, NumPy 2.5.3 y Matplotlib 3.11.2. Las nueve celdas de código quedaron guardadas con contadores consecutivos del 1 al 9, todas las comprobaciones internas terminaron correctamente y se generaron cuatro figuras.

Archivos revisados:

- [Enunciado del entregable](./Enunciado_entregable_1.pdf).
- [Notebook final](../Notebooks/1_Algebra_Lineal_6_Entregable_1_PCA_PDM.ipynb).
- [Script de transformación](./Python/transformar_BBDD.py).
- [Script de alineamiento](./Python/Alinear_landmarks.py).
- Base local `imm_face_db`, compuesta por 240 pares de ficheros `.asf` y `.jpg`.

## Qué pide el entregable y dónde se resuelve

| Requisito | Implementación | Resultado |
|---|---|:---:|
| Leer las imágenes y los landmarks | `read_asf` interpreta los `.asf`, asocia cada imagen y reconstruye los siete contornos | Cumple |
| Mostrar una cara anotada | Se superponen los 58 landmarks sobre una imagen de la base | Cumple |
| Alinear todas las formas | Procrustes generalizado elimina traslación, escala y rotación | Cumple |
| Mostrar antes y después | Figura comparativa con las 240 formas originales y alineadas | Cumple |
| Construir los vectores y la covarianza | Se usa el orden `x1,…,xn,y1,…,yn` y el divisor `N` indicado en el PDF | Cumple |
| Aplicar la EVD/PCA | `numpy.linalg.eigh` obtiene autovalores y autovectores, ordenados de mayor a menor | Cumple |
| Dibujar autovalores y suma acumulada | Ambas curvas aparecen en la misma figura | Cumple |
| Generar nueve instancias del PDM | Rejilla 3 × 3: tres modos y los valores `−3√λ`, `0`, `+3√λ` | Cumple |
| Verificar el resultado | Nueve comprobaciones algebraicas y geométricas terminan correctamente | Cumple |

## Resolución paso a paso

### 1. Lectura y control de la base de datos

Los ficheros `.asf` contienen coordenadas relativas de los puntos faciales, su pertenencia a un contorno y la información necesaria para saber si el contorno es abierto o cerrado. La función `read_asf` adapta la lógica de `getstructure` de `transformar_BBDD.py` y devuelve para cada muestra:

- los 58 puntos como una matriz de tamaño `58 × 2`;
- siete intervalos que identifican cejas, ojos, nariz, boca y mandíbula;
- siete indicadores de contorno abierto o cerrado;
- la ruta de la imagen correspondiente.

Antes de calcular nada se comprueba que existen exactamente 240 pares `.asf`/`.jpg`, que los nombres coinciden, que hay 40 sujetos, que todas las formas tienen 58 landmarks, que las coordenadas son finitas y están en el intervalo relativo `[0,1]`, y que todas las muestras comparten la misma topología. También se verifica la resolución real de las imágenes.

La primera figura transforma las coordenadas relativas en píxeles y superpone los contornos sobre una imagen. Esta comprobación visual confirma que el lector interpreta correctamente los datos.

### 2. Alineamiento Procrustes generalizado

El PCA debe medir cambios de forma, no diferencias debidas a la posición, el tamaño o la rotación de una cara. Por eso cada conjunto de landmarks se centra y se alinea contra una referencia común.

Para una forma centrada `X` y una referencia centrada `Y`, se calcula la descomposición SVD de `XᵀY`. A partir de ella se obtiene la rotación propia que minimiza la distancia entre ambas formas; si la rotación implicara una reflexión se corrige su signo. Después se estima la escala óptima mediante

```text
s = <XR,Y> / <X,X>.
```

El procedimiento se repite para las 240 formas, se calcula una nueva forma media normalizada y se itera hasta que el cambio de la referencia es menor que `10⁻¹²`. El algoritmo converge en siete iteraciones; el cambio final es `2.669 × 10⁻¹⁴` y el error máximo de centrado es `3.996 × 10⁻¹⁶`.

La segunda figura muestra por qué este paso es necesario: antes del alineamiento domina la dispersión de posición y escala; después, los puntos quedan en un sistema común y la variación observable corresponde principalmente a la geometría facial.

### 3. Matriz de datos y covarianza

Cada forma alineada se vectoriza exactamente en el orden exigido por el enunciado:

```text
z = [x₁,…,x₅₈,y₁,…,y₅₈]ᵀ.
```

Al apilar las 240 muestras se obtiene una matriz `Z` de tamaño `240 × 116`. Su media es

```text
μ̂ = (1/N) Σᵢ zᵢ,
```

y la matriz de covarianza utilizada es

```text
R̂ = (1/N) Σᵢ (zᵢ − μ̂)(zᵢ − μ̂)ᵀ
   = Zcᵀ Zc / N.
```

Por tanto, `R̂` tiene tamaño `116 × 116`. Se usa expresamente el divisor `N = 240`, no `N − 1`, porque esa es la definición dada en el PDF.

### 4. EVD y componentes principales

Como `R̂` es real y simétrica, se utiliza `numpy.linalg.eigh`, que es la rutina adecuada para este tipo de matrices:

```text
R̂ = U Λ Uᵀ.
```

Los autovalores y las columnas de `U` se ordenan conjuntamente de mayor a menor. Los pequeños autovalores negativos del orden de `10⁻²⁰` se conservan para las verificaciones algebraicas, pero se recortan a cero únicamente al calcular porcentajes y raíces cuadradas, pues proceden del redondeo en coma flotante.

La tercera figura contiene tanto el espectro de autovalores como su suma acumulada. También incorpora una escala porcentual secundaria y señala que hacen falta 22 modos para alcanzar el 95 % de la varianza.

### 5. Generación del Point Distribution Model

El modelo genera una forma mediante

```text
z̃ = μ̂ + U b.
```

Para estudiar el modo `i` se dejan todos los coeficientes a cero salvo `bᵢ`. La implementación equivalente utilizada es

```text
z̃ = μ̂ + c √λᵢ uᵢ,    c ∈ {−3, 0, 3},
```

donde `uᵢ` es la columna `i` de `U`. Se aplica a los tres autovalores principales y se obtienen exactamente nueve formas. Después cada vector de 116 componentes vuelve a convertirse en una matriz `58 × 2`.

La cuarta figura organiza el resultado en una rejilla `3 × 3`:

- cada fila representa uno de los tres primeros modos;
- las columnas representan `−3√λᵢ`, `0` y `+3√λᵢ`;
- la columna central es la forma media en las tres filas;
- todos los paneles comparten límites y proporción, para no distorsionar la comparación.

El signo de un autovector es matemáticamente arbitrario. Por ello otro programa correcto podría intercambiar visualmente las columnas negativa y positiva de un modo sin cambiar el modelo ni el resultado algebraico.

## Resultados numéricos principales

| Magnitud | Resultado |
|---|---:|
| Formas alineadas | `240 × 58 × 2` |
| Matriz de datos | `240 × 116` |
| Matriz de covarianza | `116 × 116` |
| Primer autovalor | `0.008196430849` |
| Segundo autovalor | `0.002993664115` |
| Tercer autovalor | `0.001139577993` |
| Varianza acumulada en tres modos | `70.1056107 %` |
| Modos para 95 % / 99 % | `22 / 55` |
| Amplitudes `3√λ₁`, `3√λ₂`, `3√λ₃` | `0.271602426`, `0.164143160`, `0.101272908` |
| Rango numérico de la covarianza | `113` |

## Uso de los scripts suministrados

Los scripts se han usado como referencia para conservar el formato de datos, la separación de contornos y la idea del alineamiento iterativo. No se ejecutan literalmente porque contienen supuestos que no son adecuados para un notebook reproducible en este repositorio:

- `transformar_BBDD.py` descarga de nuevo una base que ya está disponible y usa la ruta Unix fija `/tmp/imm_face_db/`;
- ambos scripts ejecutan órdenes específicas de IPython y el alineador fuerza el backend gráfico `qt5`;
- `Alinear_landmarks.py` normaliza la nueva media con `x_bar[0]`, es decir, sólo con el primer landmark, en lugar de usar la norma de la forma completa;
- el ángulo calculado con `arctan(b/a)` es sensible a división por cero y pierde información de cuadrante;
- el bucle original no tiene un límite máximo de iteraciones.

El notebook mantiene el objetivo matemático del código original y sustituye esos puntos por lectura local, rutas portables con `pathlib`, norma de Frobenius completa, rotación mediante SVD, control de reflexiones, tolerancia explícita y un máximo de iteraciones.

## Revisión de código innecesario y valores fijados

Tras la revisión se eliminó una constante de color que no se utilizaba y la resolución mostrada en la tabla pasó a derivarse de los archivos leídos. No quedan imports, funciones ni variables completas sin uso.

Los valores `240`, `58`, `40`, `116`, `640 × 480` y siete contornos aparecen en aserciones. Esto es intencional: son propiedades del conjunto de datos definido por el enunciado y sirven para detectar una copia incompleta o alterada. No intervienen para fabricar los autovalores, la covarianza ni las formas generadas. Los resultados numéricos resumidos en las celdas Markdown tampoco se usan en los cálculos; documentan la ejecución sobre la base fija del ejercicio.

No hay rutas personales absolutas, descargas, aleatoriedad, archivos temporales ni estado oculto entre celdas. El flujo es lineal y determinista. La única condición de ubicación es que el kernel se inicie desde la raíz del repositorio o desde una carpeta descendiente, como `Algebra 1/Notebooks`, para que la búsqueda ascendente encuentre los datos.

## Validación técnica

La última ejecución guardada produjo:

- contadores de ejecución consecutivos `[1,2,3,4,5,6,7,8,9]`;
- cero salidas de error;
- cuatro figuras PNG;
- un documento `nbformat` válido;
- nueve comprobaciones internas aprobadas.

Las comprobaciones cubren la simetría de la covarianza, el orden y la no negatividad numérica de los autovalores, la ortonormalidad de los autovectores, dos identidades de varianza, la reconstrucción con todos los modos, la igualdad de las tres formas centrales y la simetría de los extremos respecto a la media.

| Comprobación | Error observado |
|---|---:|
| Ortonormalidad de `U` | `1.826 × 10⁻¹⁴` |
| Igualdad traza–suma de autovalores | `1.041 × 10⁻¹⁷` |
| Reconstrucción relativa con todos los modos | `1.903 × 10⁻¹⁶` |
| Coincidencia de las formas centrales | `0` |
| Simetría de los extremos | `5.259 × 10⁻¹⁷` |

### Calidad del artefacto

| Categoría | Defectos observados | Valoración |
|---|---:|---|
| Requisitos agrupados del enunciado | 0/9 | Los nueve bloques de la matriz de trazabilidad están implementados |
| Celdas de código | 0/9 | Ejecución ordenada, sin errores ni dependencia de estado previo |
| Figuras | 0/4 | Legibles, con escalas coherentes y sin elementos omitidos |
| Símbolos importados y funciones | 0/19 | Los nueve símbolos importados y las diez funciones están utilizados |

### Correctitud analítica y robustez

| Categoría | Defectos observados | Valoración |
|---|---:|---|
| Cálculos principales | 0/8 | Lectura, alineamiento, vectorización, media, covarianza, EVD, ordenación y PDM reproducidos |
| Comprobaciones numéricas | 0/9 | Todas terminan en verdadero con tolerancias estrictas |
| Grupos de resultados contrastados | 0/8 | Dimensiones, convergencia, autovalores, varianza, umbrales, amplitudes, rango y errores coinciden |
| Relaciones entre figuras y datos | 0/4 | Las cuatro figuras usan directamente los arrays calculados |

Los cocientes `0/N` anteriores significan que no se observó ningún defecto entre las `N` unidades aplicables revisadas; no expresan una probabilidad ni una garantía fuera de este conjunto de datos y este entorno.

## Cómo reproducir la ejecución

Desde PowerShell, en la raíz del repositorio:

```powershell
.\.venv\Scripts\Activate.ps1
python -m jupyter notebook
```

Después se abre `Algebra 1/Notebooks/1_Algebra_Lineal_6_Entregable_1_PCA_PDM.ipynb` y se selecciona **Run All**. También puede ejecutarse y guardarse directamente con:

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=600 "Algebra 1\Notebooks\1_Algebra_Lineal_6_Entregable_1_PCA_PDM.ipynb"
```

## Conclusión

El entregable está resuelto correctamente. La implementación reproduce los cinco pasos descritos en el enunciado, genera las nueve caras solicitadas y presenta los autovalores junto con su suma acumulada. La última revisión no ha encontrado defectos matemáticos o funcionales pendientes. Las constantes específicas del dataset se usan como controles de integridad, no como atajos para producir el resultado.
