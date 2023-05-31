import numpy as np
import cv2 as cv
from pathlib import Path


def get_image():
    Class = 'W'  # Nome da classe para criar a pasta correspondente no diretório "DATASET"
    Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True)  # Cria a pasta se ainda não existir
    
    cap = cv.VideoCapture(0)  # Inicializa a captura de vídeo da câmera
    if not cap.isOpened():
        print("Não é possível abrir a câmera")
        exit()
    
    i = 0  # Contador de imagens capturadas
    while True:
        ret, frame = cap.read()  # Captura um frame do vídeo
        
        if not ret:
            print("Não é possível receber o quadro (fim do fluxo?). Saindo...")
            break
        
        i += 1
        if i % 5 == 0:  # A cada 5 frames capturados, salva a imagem
            cv.imwrite('DATASET/'+Class+'/'+str(i)+'.png', frame)
      
        cv.imshow('quadro', frame)  # Mostra o frame capturado na janela "frame"
        
        if cv.waitKey(1) == ord('q') or i > 500:  # Sai do loop quando pressionada a tecla 'q' ou captura mais de 500 imagens
            break
  
    cap.release()  # Libera a câmera
    cv.destroyAllWindows()  # Fecha todas as janelas abertas

if __name__ == "__main__":
    get_image()
