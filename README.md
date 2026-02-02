# multiagente-a2a-saas

Pequeño framework de agentes A2A (almacenador, analizadores y orquestador) — pruebas y utilidades incluidas.

## 🚀 Qué hay en este repo
- `almacenador_agent/`, `analisador_agent/`, `orquestador_agent/` — agentes y scripts de prueba.
- Tests simples ejecutables en cada agente (`test.py`).
- Código pensado para integrarse con Qdrant y A2A SDKs.

## ✅ Preparación rápida (Windows / PowerShell)
1. Crear y activar entorno virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Instalar dependencias:
   ```powershell
   pip install -r requirements.txt
   pip install pytest
   ```
3. Ejecutar los tests de los agentes:
   ```powershell
   python almacenador_agent/test.py
   python analisador_agent/test.py
   python orquestador_agent/test.py
   ```

## 🔧 Variables de entorno importantes
Crea un archivo `.env` (no subir al repo) con, al menos:
- OPENROUTER_API_KEY
- QDRANT_URL
- QDRANT_API_KEY
- COLLECTION_NAME

## 🧪 CI
Se incluye un workflow de GitHub Actions que ejecuta los tests en Python 3.9–3.11.

## 🔁 Flujo sugerido para subir a GitHub
```powershell
git init
git add .
git commit -m "chore: prepare repository for GitHub (README, CI, LICENSE, .gitignore)"
# Usando GitHub CLI (recomendado):
gh repo create my-org/my-repo --public --source=. --remote=origin --push
# O manualmente:
# git remote add origin <url>
# git push -u origin main
```

## 📄 Licencia
Este repositorio añade por defecto la licencia **MIT** (archivo `LICENSE`). Cámbiala si prefieres otra.

## ➕ Próximos pasos recomendados
1. Revisar y completar el `.env` con las claves reales. ⚠️ No subir claves al repo.
2. Elegir la licencia (MIT por defecto) y actualizar el `LICENSE` con tu nombre.
3. Crear el repo en GitHub y habilitar Actions.

---
Si quieres, puedo: 1) crear y commitear el repo localmente, o 2) ejecutar los comandos para crear el repo remoto (necesitaré tu confirmación).