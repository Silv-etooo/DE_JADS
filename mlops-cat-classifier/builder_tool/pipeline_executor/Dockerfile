FROM python:3.10.7-slim
ADD pipeline_executor.py /home/pipeline_executor.py
ADD requirements.txt ./
RUN pip install -r requirements.txt
COPY pipeline_executor.py .
ENTRYPOINT ["python","/home/pipeline_executor.py"]