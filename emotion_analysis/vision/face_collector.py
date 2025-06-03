import cv2
import os
from pathlib import Path
import click
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class FaceCollector:
    def __init__(self, video_path, output_dir, model_path, config):
        self.video_path = video_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.config = config
        self.face_detector = YOLO(model_path)
        
        # Criar diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
    def detect_faces(self, frame):
        # Detectar e rastrear faces usando YOLOv8
        results = self.face_detector.track(frame, persist=True, verbose=False)
        
        faces = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box, track_id in zip(boxes, boxes.id):
                # Obter coordenadas do box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                if confidence > self.config['faces']['collection']['confidence_threshold']:
                    faces.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'id': int(track_id),
                        'confidence': float(confidence)
                    })
        
        return faces
    
    def collect_faces(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        faces_collected = defaultdict(int)  # Contador de faces por ID
        
        click.echo("Iniciando coleta de faces...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Processar a cada N frames para não sobrecarregar
            if frame_count % self.config['faces']['collection']['save_interval'] == 0:
                faces = self.detect_faces(frame)
                
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    track_id = face['id']
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Verificar tamanho mínimo
                    if face_img.shape[0] < self.config['faces']['collection']['min_face_size'] or \
                       face_img.shape[1] < self.config['faces']['collection']['min_face_size']:
                        continue
                    
                    # Criar diretório para o ID se não existir
                    id_dir = os.path.join(self.output_dir, f"person_{track_id}")
                    os.makedirs(id_dir, exist_ok=True)
                    
                    # Salvar face
                    face_num = faces_collected[track_id] + 1
                    face_filename = f"{face_num}.jpg"
                    face_path = os.path.join(id_dir, face_filename)
                    
                    cv2.imwrite(face_path, face_img)
                    faces_collected[track_id] += 1
                    
                    click.echo(f"Face coletada: person_{track_id}/{face_filename}")
            
            # Mostrar progresso a cada 100 frames
            if frame_count % 100 == 0:
                click.echo(f"Frames processados: {frame_count}")
        
        cap.release()
        
        # Atualizar configuração com as faces coletadas
        self.config['faces']['groups'] = {}
        for track_id, count in faces_collected.items():
            group_key = f"person_{track_id}"
            self.config['faces']['groups'][group_key] = {
                'name': f"Pessoa {track_id}",
                'faces': [f"{i}.jpg" for i in range(1, count + 1)]
            }
        
        click.echo("\nColeta finalizada!")
        for track_id, count in faces_collected.items():
            click.echo(f"person_{track_id}: {count} faces coletadas")
        
        return faces_collected 