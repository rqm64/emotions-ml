# Emotions ML

Распознавание эмоций по лицу (Keras + FastAPI).

Dataset: https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset

## UI

```bash
cd ui
npm install
npm run dev
```

Откройте http://localhost:5173 — кнопка «Старт» по центру, после нажатия полноэкранное видео с камеры (нужен HTTPS или localhost).
UI раз в ~1.8с отправляет кадр на сервер, получает распознанную эмоцию и вероятности, и отображает поверх видео анимированный смайлик + топ эмоций.

## Сервер

### Установка

```bash
cd server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Запуск

```bash
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Документация API: http://localhost:8000/docs

### Endpoint предикта

`POST /predict` принимает `multipart/form-data` с полем `image` (файл изображения), возвращает:

```json
{
  "emotion": "happy",
  "probability": 0.9132,
  "top_predictions": [
    { "emotion": "happy", "probability": 0.9132 },
    { "emotion": "surprise", "probability": 0.0481 },
    { "emotion": "neutral", "probability": 0.0215 }
  ]
}
```

### Датасет

Положите изображения в `server/data/train/` — по одной подпапке на класс:

```
server/data/train/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/
```

### Обучение (локально, без API)

```bash
cd server
source .venv/bin/activate
python train.py --epochs 10
```

Сохраняется в `server/artifacts/model.keras` и `server/artifacts/metadata.json`.
