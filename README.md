
## ⚙️ Установка

### 1. Клонируй репозиторий

```bash
git clone <repo_url>
cd vk_rag_bot
```

### 2. Создай виртуальное окружение

```bash
python -m venv venv

venv\Scripts\activate
```

### 3. Установи зависимости

```bash
pip install -r requirements.txt
```

### 4. Установи и запусти Ollama

```bash
ollama serve  
ollama pull qwen2.5-coder:3b

# Альтернативы если мало RAM:
# ollama pull qwen2.5-coder:1.5b   # ~1 ГБ, быстрее, чуть хуже
# ollama pull deepseek-coder:1.3b  # ~800 МБ, очень лёгкая
# ollama pull phi3:mini            # универсальная, 3.8B
```
### 6. Создай файл `.env` 
VK_TOKEN=your_vk_token_here
VK_GROUP_ID=123456789
---
