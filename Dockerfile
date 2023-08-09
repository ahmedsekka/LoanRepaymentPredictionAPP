FROM python:3.8

WORKDIR /app

COPY functions2.py corr.py requirements.txt lgbmfitt.joblib LGBM_train_fit.joblib ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "corr.py"]
