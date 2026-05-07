import base64
import json
import requests

# === НАСТРОЙКИ ===
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "maternion/LightOnOCR-2:latest"  # или "qwen2.5-vl" если glm-ocr падает
IMAGE_PATH = "2.jpg"    # укажи путь к своей УПД
# =================

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    image_b64 = encode_image(IMAGE_PATH)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "images": [image_b64]  # base64 массив — критично для vision-моделей
            }
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384,      # glm-ocr требует большой контекст
            "temperature": 0.0,    # минимум галлюцинаций
            "num_predict": 4096    # хватит на большую таблицу
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # Вывод ответа модели
        print("=== РЕЗУЛЬТАТ ===")
        print(result["message"]["content"])
        
        # Для отладки: можно посмотреть сколько токенов ушло
        print(f"\n[DEBUG] Prompt tokens: {result.get('prompt_eval_count', 'N/A')}")
        print(f"[DEBUG] Response tokens: {result.get('eval_count', 'N/A')}")

    except requests.exceptions.ConnectionError:
        print("❌ Ollama не запущен. Запусти: ollama serve")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Ошибка HTTP: {e}")
        print(f"Ответ сервера: {e.response.text}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()