# Medical Chatbot

python 3.10

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install accelerate, peft, bitsandbytes, transformers, trl, flask

docker build -t shibingchat:v0 .
docker run --name shibingchat --gpus all -p 5000:5000 -it shibingchat:v0