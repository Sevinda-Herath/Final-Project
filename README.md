pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

"url for port 8000"/predict?symbol=TSLA