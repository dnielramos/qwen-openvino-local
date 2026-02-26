# Qwen 2.5 + OpenVINO: Chat de Alta Velocidad (Local)

¡Bienvenido al futuro de la inferencia local! Este proyecto te permite ejecutar el modelo de lenguaje **Qwen 2.5 (3B)** directamente en tu hardware Intel usando **OpenVINO**. Todo el framework está optimizado para conseguir un rendimiento espectacular y cero latencias aprovechando tu Tarjeta Gráfica (Intel iGPU/Arc).

El proyecto consta de dos partes principales perfectamente integradas:
1.  **Backend (Python + FastAPI)**: Servidor robusto que carga el modelo en la memoria de la GPU y utiliza WebSockets para emitir los tokens generados a medida que los piensa la IA.
2.  **Frontend (HTML + JS + Tailwind CSS)**: Interfaz de usuario "Premium" responsivo, en Modo Oscuro y elegante. Dispone de un renderizado perfecto a 60 FPS con estilo "máquina de escribir" (throttling + WebSockets), soporte de Markdown y resaltado de código.

---

## 🚀 Requisitos Previos

Necesitarás tener instalados:
-   **Python 3.10 o superior**.
-   **Hardware Intel compatible con OpenVINO** (iGPU integrada o dedicada Arc).
-   El modelo **Qwen2.5-3B-Instruct-Q4_K_M.gguf** (colocado en el directorio `model_cache/qwen25_3b_gguf/` relativo al script de python).

---

## 🛠️ Instalación y Configuración Paso a Paso

Sigue esta guía para poner en marcha el proyecto en pocos minutos.

### 1. Clonar y Preparar el Entorno Virtual
Abre tu terminal (PowerShell o CMD) en la carpeta donde deseas guardar el proyecto y ejecuta:

```bash
# Crear un entorno virtual para aislar dependencias
python -m venv venv

# Activar el entorno virtual (Windows)
.\venv\Scripts\activate
```

Tu terminal debería mostrar ahora `(venv)` al inicio de la línea.

### 2. Instalar Dependencias
Una vez activo el entorno, instala todas las dependencias necesarias de nuestro archivo de requerimientos:

```bash
pip install -r requirements.txt
```

> **Nota para OpenVINO:** Este comando instalará paquetes críticos como `fastapi`, `uvicorn[standard]` (para WebSockets de alto rendimiento), y `openvino_genai`.

### 3. Verificar el Modelo
Asegúrate de haber descargado el formato optimizado del modelo en GGUF. La ruta exacta en la que el backend espera encontrar el archivo es:
`tu-proyecto/model_cache/qwen25_3b_gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf`

---

## ▶️ Ejecución y Uso de la Aplicación

¡Hora de la magia! Sigue estos dos sencillos pasos.

### Paso A: Levantar el Servidor de IA (Backend)
Con el entorno virtual activado, inicia el servidor de FastAPI:

```bash
python qwen3b-GGUF-Q4_K_M.py
```

Verás una salida similar a esta en consola:
```text
INFO:__main__:🚀 Inicializando motor de IA...
INFO:__main__:⚡ Intentando cargar modelo en GPU (Intel iGPU/Arc)...
INFO:__main__:✅ ¡ÉXITO! Modelo cargado en GPU. El rendimiento será máximo.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
🔹 *¡No cierres esta ventana de terminal mientras uses la App!*

### Paso B: Abrir la Interfaz de Usuario (Frontend)
1. Busca el archivo **`main.html`** en tu organizador de archivos.
2. Haz **Doble Clic** para abrirlo en tu navegador favorito (Chrome, Edge, Firefox, etc.).
3. ¡Escribe un mensaje de saludo o escríbele a Qwen que te redacte código Python! 

Disfrutarás de una visualización inmersiva en la que el mensaje se forma palabra por palabra a una fluidez de 60 FPS, sin ningún congelamiento de pantalla.

---

## 🧪 Pruebas de Sistema (Opcional)

Si alguna vez necesitas validar que el backend y la conexión WebSocket funcionan perfectamente *sin abrir el navegador*, puedes usar el script de prueba integrado.

En una **nueva terminal** (con el entorno `venv` activado), ejecuta:
```bash
python test_ws.py
```
Este script se conectará al puerto local 8000, solicitará un mensaje corto a la IA y validará la consistencia del *Streaming* en la consola.

---

*Desarrollado Por Ingeniero Daniel Ramos con ❤️ para máxima velocidad de inferencia Local.*
