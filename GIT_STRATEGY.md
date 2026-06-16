# Estrategia de Git — Churn Prediction LLM

## Flujo utilizado: GitHub Flow

Este proyecto sigue **GitHub Flow**, una estrategia simple y
efectiva para proyectos con ciclos de entrega continuos.

## Ramas

| Rama | Propósito |
|---|---|
| `main` | Rama estable. Solo recibe cambios vía Pull Request aprobado. Refleja siempre la versión en producción. |
| `development` | Rama de trabajo diario. Aquí se desarrollan y prueban todos los cambios antes de integrarse a `main`. |

## Flujo de trabajo

main ←── Pull Request ←── development
↑
commits diarios
de desarrollo

1. Todo el trabajo se realiza en la rama `development`.
2. Los commits siguen la convención **Conventional Commits**:
   - `feat:` nueva funcionalidad
   - `fix:` corrección de errores
   - `chore:` tareas de mantenimiento (estructura, dependencias)
   - `docs:` cambios en documentación
3. Cuando el trabajo está completo y probado, se abre un
   **Pull Request** de `development` → `main` describiendo
   los cambios realizados.
4. El PR se revisa, se aprueba y se fusiona con **merge commit**
   (no squash, para preservar el historial).
5. Se crea un **Release** en GitHub con tag semántico (`v1.0.0`)
   documentando las funcionalidades incluidas.

## Convención de commits

tipo: descripción breve en presente e imperativo
Ejemplos:
feat: agrega notebook de preprocesamiento con EDA
fix: corrige conversión de TotalCharges a numérico
chore: actualiza requirements.txt con nuevas dependencias
docs: completa README con model card y resultados

## Historial de releases

| Versión | Descripción |
|---|---|
| `v1.0.0` | Primera versión del proyecto: preprocesamiento, 3 modelos ML, tuning, SHAP y explicador LLM |