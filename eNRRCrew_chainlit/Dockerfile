FROM python:3.12.7
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt 
RUN pip install graphrag==0.3.6 --no-dependencies --ignore-installed
COPY . .
EXPOSE 8080
CMD ["python", "-m", "chainlit", "run", "appUI.py", "-h", "--port", "8080", "--host", "0.0.0.0"]

