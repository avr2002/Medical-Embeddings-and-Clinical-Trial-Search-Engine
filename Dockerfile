FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r src/requirements.txt
EXPOSE $PORT
CMD streamlit run src/app.py  