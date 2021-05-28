FROM python:3.8-buster
ADD requirements.txt /app/requirements.txt
WORKDIR /app/
RUN pip install -r requirements.txt
RUN pip install 'ray[default]'
RUN ray start --head
CMD ["python", "./tuning.py"]

