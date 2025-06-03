import click
import yaml
import cv2
import os
from pathlib import Path
from emotion_analysis.vision.capture import VideoCapture
from emotion_analysis.vision.detection import EmotionDetector
from emotion_analysis.io.logging import EmotionEventLogger
from emotion_analysis.vision.processing import VideoProcessor
from emotion_analysis.vision.video_write import VideoWriter
from emotion_analysis.vision.face_collector import FaceCollector
from emotion_analysis.vision.face_labeler import FaceLabeler
from emotion_analysis.vision.face_recognizer import FaceRecognizer
from flask_opencv_streamer.streamer import Streamer

def load_config(config_path):
    """Carrega o arquivo de configuração YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_known_faces(faces_dir):
    """Carrega as faces conhecidas do diretório especificado."""
    known_faces = []
    if os.path.exists(faces_dir):
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(faces_dir, filename)
                known_faces.append({
                    'name': name,
                    'path': image_path
                })
    return known_faces

@click.group()
def cli():
    """Sistema de Análise de Emoções em Vídeo."""
    pass

@cli.command()
@click.option('--config', '-c', 
              default='config.yaml',
              help='Caminho para o arquivo de configuração YAML',
              type=click.Path(exists=True))
@click.option('--input', '-i',
              help='Sobrescreve o caminho do vídeo de entrada do config.yaml',
              type=click.Path(exists=True))
@click.option('--output', '-o',
              help='Sobrescreve o caminho do vídeo de saída do config.yaml',
              type=click.Path())
@click.option('--model', '-m',
              help='Sobrescreve o caminho do modelo do config.yaml',
              type=click.Path(exists=True))
@click.option('--faces-dir', '-f',
              help='Sobrescreve o diretório de faces cadastradas do config.yaml',
              type=click.Path(exists=True))
def analyze(config, input, output, model, faces_dir):
    """Analisa emoções em um vídeo."""
    # Carrega configuração
    config_data = load_config(config)
    
    # Sobrescreve configurações com argumentos da CLI se fornecidos
    if input:
        config_data['video']['input'] = str(input)
    if output:
        config_data['video']['output'] = str(output)
    if model:
        config_data['model']['path'] = str(model)
    if faces_dir:
        config_data['faces']['directory'] = str(faces_dir)

    # Carrega faces conhecidas se habilitado
    if config_data['faces']['enabled']:
        known_faces = load_known_faces(config_data['faces']['directory'])
        config_data['faces']['known_faces'] = known_faces
        click.echo(f"Faces cadastradas encontradas: {len(known_faces)}")
        for face in known_faces:
            click.echo(f"- {face['name']}")

    # Inicializa componentes
    capture = VideoCapture(config_data['video']['input'])
    frame_width = int(capture.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector = EmotionDetector(
        config_data['model']['path'],
        known_faces=config_data['faces']['known_faces'] if config_data['faces']['enabled'] else None
    )
    logger = EmotionEventLogger()
    writer = VideoWriter(
        config_data['video']['output'],
        frame_width,
        frame_height,
        fps=config_data['video']['fps']
    )
    streamer = Streamer(
        config_data['server']['port'],
        config_data['server']['debug']
    )

    # Processa o vídeo
    click.echo("Iniciando processamento do vídeo...")
    processor = VideoProcessor(capture, detector, logger, writer, streamer)
    processor.process_video()

    # Exibe resultados
    click.echo("\nEventos de emoção detectados:")
    for event in logger.get_events():
        click.echo(f"Timestamp: {event['timestamp']} - Emotion: {event['emotion']}")

@cli.command()
@click.option('--config', '-c', 
              default='config.yaml',
              help='Caminho para o arquivo de configuração YAML',
              type=click.Path(exists=True))
@click.option('--input', '-i',
              help='Sobrescreve o caminho do vídeo de entrada do config.yaml',
              type=click.Path(exists=True))
@click.option('--output-dir', '-o',
              help='Diretório para salvar as faces coletadas',
              type=click.Path())
@click.option('--model', '-m',
              help='Sobrescreve o caminho do modelo do config.yaml',
              type=click.Path(exists=True))
def face_collector(config, input, output_dir, model):
    """Coleta faces de um vídeo para treinamento."""
    # Carrega configuração
    config_data = load_config(config)
    
    # Sobrescreve configurações com argumentos da CLI se fornecidos
    if input:
        config_data['video']['input'] = str(input)
    if model:
        config_data['model']['path'] = str(model)
    
    # Define diretório de saída
    if not output_dir:
        output_dir = config_data['faces']['directory']
    
    # Inicializa coletor de faces
    collector = FaceCollector(
        video_path=config_data['video']['input'],
        output_dir=output_dir,
        model_path=config_data['model']['path'],
        config=config_data
    )
    
    # Inicia coleta
    collector.collect_faces()

@cli.command()
@click.option('--config', '-c', 
              default='config.yaml',
              help='Caminho para o arquivo de configuração YAML',
              type=click.Path(exists=True))
def label_faces(config):
    """Rotula faces coletadas interativamente."""
    labeler = FaceLabeler(config)
    labeler.interactive_labeling()

@cli.command()
@click.option('--config', '-c', 
              default='config.yaml',
              help='Caminho para o arquivo de configuração YAML',
              type=click.Path(exists=True))
def list_faces(config):
    """Lista todas as faces rotuladas."""
    labeler = FaceLabeler(config)
    labeler.list_labels()

@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Caminho para o arquivo de configuração')
@click.option('--input', '-i', help='Caminho do vídeo de entrada (sobrescreve config.yaml)')
@click.option('--output', '-o', help='Caminho do vídeo de saída (sobrescreve config.yaml)')
@click.option('--model', '-m', help='Caminho do modelo YOLO (sobrescreve config.yaml)')
def face_recognition(config, input, output, model):
    """Reconhecimento facial em vídeo"""
    # Carregar configuração
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Sobrescrever configurações se fornecidas
    if input:
        config_data['video']['input'] = input
    if output:
        config_data['video']['output'] = output
    if model:
        config_data['model']['path'] = model
    
    # Inicializar reconhecedor
    recognizer = FaceRecognizer(
        video_path=config_data['video']['input'],
        output_path=config_data['video']['output'],
        model_path=config_data['model']['path'],
        config=config_data
    )
    
    # Processar vídeo
    recognizer.process_video()

if __name__ == "__main__":
    cli()
