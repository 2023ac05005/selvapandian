
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt &&     python - <<'PY'import nltk
try:

    nltk.download('punkt')

    nltk.download('punkt_tab')

except Exception as e:

    print('NLTK download warning:', e)
PY

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/ui_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
