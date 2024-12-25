# Reasoning Grounding DINO to Segment Anything
## Environment setting

### Python requiremnet
python version required: `python>=3.11` in linux

### Install Ollama

**Hardware Requirements**

- **CPU**: Multicore processor

- **RAM**: Minimum of 16 GB recommended

- **GPU**: NVIDIA RTX series (for optimal performance), at least 8 GB VRAM

**Step1**:
Download ollama from this site according to your operating system
https://ollama.com/download/linux
```
curl -fsSL https://ollama.com/install.sh | sh
```
**Step2**:
open your teminal

**Step3**:
run following commands in your terminal
```
ollama serve
ollama pull llama3.2:3b  & ollama pull nomic-embed-text
```

### Environment Building
Install requirement with below:
```
pip install -e .
```
## Usage
To load all model on local, use the command below:
```
python main.py
```
For `main_pro.py`, you need to go [here](https://cloud.deepdataspace.com/docs#/api/grounding_dino) to get the api token and put in `main_pro.py` first, and then you can run 
```
python main_pro.py
```