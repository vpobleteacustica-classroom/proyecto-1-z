# Informe de Evaluación – Hitos 2 y 3  
**Curso:** ACUS220 
**Proyecto:** Sistema de transcripción inteligente de guitarra  
**Grupo:** Transcripción Guitarra  

---

## 1. Descripción general del proyecto

El proyecto aborda el problema de la **transcripción automática de guitarra a partir de grabaciones polifónicas**, combinando técnicas modernas de **separación de fuentes** y **transcripción musical**. El sistema integra modelos de deep learning preentrenados (Demucs, MDX Guitar y Basic Pitch) para aislar la pista de guitarra desde audio completo y generar una representación simbólica en formato MIDI.

La propuesta es ambiciosa, técnicamente actual y bien alineada con el estado del arte en **Music Information Retrieval (MIR)** y **Automatic Music Transcription (AMT)**, destacando por su correcta articulación entre modelos existentes y un pipeline funcional de procesamiento.

---

## 2. Evaluación por criterios (Hitos 2 y 3)

| Criterio | Peso | Nota | Comentario |
|--------|------|------|------------|
| Claridad del planteamiento del problema | 0.15 | 7.0 | Problema claramente definido y bien motivado; el objetivo de transcripción automática de guitarra está formulado con precisión y pertinencia. |
| Justificación y contexto del experimento | 0.10 | 6.9 | Excelente revisión del estado del arte en AMT y MIR; referencias actuales y dominio conceptual sobresaliente. |
| Metodología y organización del notebook | 0.20 | 6.8 | Pipeline bien estructurado y funcional; se sugiere profundizar en la validación cuantitativa y evaluación sistemática del desempeño. |
| Calidad del código y buenas prácticas | 0.15 | 6.9 | Código limpio, legible y bien organizado; se recomienda mayor modularización y uso de docstrings. |
| Análisis de resultados y visualizaciones | 0.15 | 6.7 | Análisis interpretativo sólido; faltan ejemplos visuales adicionales (espectrogramas, comparación MIDI/audio). |
| Conclusiones y coherencia con objetivos | 0.15 | 6.9 | Conclusiones claras, bien alineadas con los objetivos; se destaca la viabilidad del sistema. |
| Redacción, ortografía y estilo general | 0.10 | 7.0 | Redacción excelente, lenguaje técnico riguroso y presentación clara. |

**Nota Hitos 2 y 3 (ponderada): 6.7**

---

## 3. Observación relevante sobre reproducibilidad

Si bien el proyecto declara la ejecución preferente en **Google Colab**, durante la revisión se observaron **errores de dependencias y conflictos de versiones** al intentar ejecutar el notebook en dicho entorno. Para un usuario externo, estos problemas pueden dificultar significativamente la reproducibilidad del sistema.

Se recomienda explícitamente:
- Detallar paso a paso la configuración exacta del entorno en Colab (versiones de Python, instalación forzada de librerías y reinicios de kernel).
- Incluir una sección clara de *troubleshooting* con errores comunes y sus soluciones.
- Proveer, idealmente, un notebook de Colab completamente funcional y verificado.

Este punto no invalida el aporte técnico del proyecto, pero sí es relevante desde el punto de vista de **usabilidad y transferencia del trabajo**.

---

## 4. Nota final del curso

- **Nota Hito 1:** 6.7  
- **Nota Hitos 2 y 3:** 6.7  

**Nota final del curso:** **6.7**

---

## 5. Comentario global de cierre

Trabajo de nivel sobresaliente que combina **rigor técnico, dominio conceptual y creatividad**, integrando de forma efectiva herramientas modernas de deep learning aplicadas a música computacional. El proyecto demuestra madurez técnica y una muy buena capacidad de aprendizaje autónomo, además de una participación destacada y constante durante el curso.

Se valora especialmente la ambición del enfoque, la correcta integración de modelos complejos y la reflexión crítica sobre las limitaciones actuales del sistema. Como proyección futura, este trabajo tiene un alto potencial de extensión hacia validaciones más profundas, fine-tuning de modelos y aplicaciones musicales reales.

**¡Felicitaciones por el excelente trabajo!**
