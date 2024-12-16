# Reasoning Grounding DINO to Segment Anything
## Environment setting
### Install Ollama
**Hardware Requirements** <br>
**CPU**: Multicore processor<br>
**RAM**: Minimum of 16 GB recommended<br>
**GPU**: NVIDIA RTX series (for optimal performance), at least 8 GB VRAM<br>
**Step1**:<br>
Download ollama from this site according to your operating system<br>
https://ollama.com/download/linux<br>
```
curl -fsSL https://ollama.com/install.sh | sh
```

**Step2**:<br>
open your teminal<br>
**Step3**:<br>
run following commands in your terminal<br>
```
ollama serve
ollama pull llama3.2:3b  & ollama pull nomic-embed-text
```

### Python requiremnet
python version required: python>=3.11
```
pip install -e .
```
## Usage
```
python main.py
```
