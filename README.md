# Medical Chatbot

This project is a medical chatbot for the University of Surrey Group 3 Deep Learning and Advanced AI.

## Requirementes

Python 3.10

CUDA 12.1

GPU that supports CUDA

Docker

## Installation

```bash
git clone https://github.com/PavloNa/MedicalChatBot.git
cd MedicalChatBot/Flask
```

## Changes

Inside /Flask/config.py

```python
OPENAI_API_KEY = '' #OPEN API READ/WRITE KEY
```
Inside /Flask/chat.py

```python
access_token = '' #HUGGING FACE READ TOKEN
model_name= ''#MODEL NAME, available at https://huggingface.co/pavlopt (ex. pavlopt/llama2-shibing-all)
```

## Running

```bash
docker build -t shibingchat:v0 .
docker run --name shibingchat --gpus all -p 5000:5000 -it shibingchat:v0
```

## Access

Access the app at [localhost:5000](http://localhost:5000/) after running the commands.

## Warnings

This project is very simple. Therefore, only submit one question at a time, wait for the answer. The prompt does not autodelete itself. This was developed for demonstration purposes only. It takes around 1-3 minutes to generate the response.