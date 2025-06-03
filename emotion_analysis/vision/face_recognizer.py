import cv2
import os
from pathlib import Path
import click
from ultralytics import YOLO
import bbox_visualizer as bbv
import numpy as np
import mediapipe as mp

class FaceRecognizer:
    def __init__(self, video_path, output_path, model_path, config):
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        self.config = config
        self.face_detector = YOLO(model_path)
        
        # Inicializar MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pontos dos olhos para EAR
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Pontos da boca
        self.MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # Configurações
        self.EAR_THRESHOLD = 0.15  # Threshold para olhos fechados
        self.MOUTH_THRESHOLD = 0.3  # Threshold para boca aberta
        
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def calculate_ear(self, landmarks, eye_points):
        """Calcula a proporção dos olhos (EAR)"""
        points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                          for point in eye_points])
        
        # Calcular distâncias verticais
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        
        # Calcular distância horizontal
        h = np.linalg.norm(points[0] - points[3])
        
        # Calcular EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_mouth_ratio(self, landmarks):
        """Calcula a proporção da boca"""
        points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                          for point in self.MOUTH])
        
        # Calcular distância vertical da boca
        vertical_dist = np.linalg.norm(points[13] - points[14])
        
        # Calcular distância horizontal da boca
        horizontal_dist = np.linalg.norm(points[0] - points[10])
        
        # Calcular razão
        mouth_ratio = vertical_dist / horizontal_dist
        return mouth_ratio
    
    def check_attention(self, landmarks):
        """Verifica se a pessoa está atenta baseado em olhos e boca"""
        # Calcular EAR para ambos os olhos
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        # Calcular razão da boca
        mouth_ratio = self.calculate_mouth_ratio(landmarks)
        
        # Verificar condições
        eyes_open = ear > self.EAR_THRESHOLD
        mouth_open = mouth_ratio > self.MOUTH_THRESHOLD
        
        # Regras de atenção
        if eyes_open and not mouth_open:
            return True  # Atento: olhos abertos e boca fechada
        else:
            return False  # Desatento: olhos fechados OU boca aberta
    
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
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        
        # Obter propriedades do vídeo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        click.echo("Iniciando processamento do vídeo...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detectar faces
            faces = self.detect_faces(frame)
            
            # Converter para RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            # Desenhar bounding boxes e nomes
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                track_id = face['id']
                
                # Obter nome do aluno do config
                person_key = f"person_{track_id}"
                if person_key in self.config['faces']['groups']:
                    name = self.config['faces']['groups'][person_key]
                else:
                    name = f"Pessoa {track_id}"
                
                # Verificar atenção
                is_attentive = False
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        # Verificar se os landmarks estão dentro do bbox
                        center_x = landmarks.landmark[1].x * width
                        center_y = landmarks.landmark[1].y * height
                        
                        if (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                            is_attentive = self.check_attention(landmarks)
                            break
                
                # Adicionar status de atenção ao nome
                status = "ATENTO" if is_attentive else "DESATENTO"
                label = f"{name}-{status}"
                
                # Escolher cor baseada no status
                color = (0, 255, 0) if is_attentive else (0, 0, 255)
                
                # Desenhar bbox e nome
                bbox = [x1, y1, x2, y2]
                frame = bbv.draw_rectangle(frame, bbox, bbox_color=color)
                frame = bbv.add_label(frame, label, bbox, text_bg_color=color)
            
            # Salvar frame
            out.write(frame)
            
            # Mostrar progresso a cada 100 frames
            if frame_count % 100 == 0:
                click.echo(f"Frames processados: {frame_count}")
        
        cap.release()
        out.release()
        
        click.echo(f"\nVídeo processado salvo em: {self.output_path}") 