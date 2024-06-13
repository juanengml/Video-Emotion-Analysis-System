# Video-Emotion-Analysis-System


Este projeto implementa um sistema capaz de analisar um vídeo, identificando mudanças de humor em pessoas, especificamente capturando momentos de riso, neutro e tristeza. Utiliza detecção de rostos, reconhecimento de emoções faciais e gera um vídeo de saída com as emoções identificadas.

## Instalação

Para instalar e usar este projeto, siga os passos abaixo:

1. **Clone o repositório:**

``` bash
   git clone https://github.com/seuusuario/emotion_analysis.git
   cd emotion_analysis
```


Instale as dependências:

2. Recomenda-se usar um ambiente virtual para isolar as dependências do projeto. Com virtualenv:

``` bash
python -m venv venv
source venv/bin/activate   # No Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Baixe o modelo YOLOv8 para detecção de rostos:

Baixe o arquivo yolov8n-face.pt do repositório:

!(Modelo YOLOv8 para detecção de rostos)[https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt]

4. Coloque o arquivo baixado na raiz do diretório emotion_analysis.

``` bash
python runner/main.py
```

## Demo

!()[output/output_model.mp4]
