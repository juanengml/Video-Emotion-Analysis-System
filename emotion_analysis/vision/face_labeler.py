import os
import yaml
import click
from pathlib import Path
import cv2
import shutil

class FaceLabeler:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.faces_dir = self.config['faces']['directory']
        self.labels = self.config['faces']['labels']
        
    def load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def list_unlabeled_faces(self):
        """Lista todas as faces que ainda não foram rotuladas."""
        labeled_faces = set()
        for faces in self.labels.values():
            labeled_faces.update(faces)
            
        all_faces = set(f for f in os.listdir(self.faces_dir) 
                       if f.endswith(('.jpg', '.jpeg', '.png')))
        
        return list(all_faces - labeled_faces)
        
    def add_label(self, face_file, person_name):
        """Adiciona um label para uma face."""
        if face_file not in os.listdir(self.faces_dir):
            click.echo(f"Erro: Arquivo {face_file} não encontrado!")
            return False
            
        if person_name not in self.labels:
            self.labels[person_name] = []
            
        if face_file not in self.labels[person_name]:
            self.labels[person_name].append(face_file)
            self.save_config()
            click.echo(f"Face {face_file} rotulada como {person_name}")
            return True
        return False
        
    def remove_label(self, face_file, person_name):
        """Remove um label de uma face."""
        if person_name in self.labels and face_file in self.labels[person_name]:
            self.labels[person_name].remove(face_file)
            self.save_config()
            click.echo(f"Label removido: {face_file} de {person_name}")
            return True
        return False
        
    def show_face(self, face_file):
        """Mostra uma face para identificação."""
        face_path = os.path.join(self.faces_dir, face_file)
        img = cv2.imread(face_path)
        if img is None:
            click.echo(f"Erro ao carregar imagem: {face_file}")
            return
            
        # Redimensionar para visualização
        height, width = img.shape[:2]
        max_height = 400
        if height > max_height:
            scale = max_height / height
            width = int(width * scale)
            height = max_height
            img = cv2.resize(img, (width, height))
            
        cv2.imshow('Face', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def interactive_labeling(self):
        """Interface interativa para rotular faces."""
        unlabeled = self.list_unlabeled_faces()
        if not unlabeled:
            click.echo("Não há faces para rotular!")
            return
            
        click.echo(f"\nFaces não rotuladas: {len(unlabeled)}")
        
        for face_file in unlabeled:
            self.show_face(face_file)
            
            # Mostrar pessoas disponíveis
            click.echo("\nPessoas disponíveis:")
            for i, person in enumerate(self.labels.keys(), 1):
                click.echo(f"{i}. {person}")
            click.echo("n. Nova pessoa")
            click.echo("s. Pular")
            
            choice = click.prompt("Escolha uma opção", type=str)
            
            if choice.lower() == 's':
                continue
            elif choice.lower() == 'n':
                new_person = click.prompt("Nome da nova pessoa")
                self.labels[new_person] = []
                self.add_label(face_file, new_person)
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.labels):
                        person = list(self.labels.keys())[idx]
                        self.add_label(face_file, person)
                    else:
                        click.echo("Opção inválida!")
                except ValueError:
                    click.echo("Opção inválida!")
                    
        click.echo("\nRotulagem concluída!")
        
    def list_labels(self):
        """Lista todas as faces rotuladas."""
        for person, faces in self.labels.items():
            if faces:
                click.echo(f"\n{person}:")
                for face in faces:
                    click.echo(f"  - {face}")
            else:
                click.echo(f"\n{person}: (sem faces)") 