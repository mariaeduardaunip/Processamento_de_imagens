import sys
import cv2
import numpy as np

# Importa as classes necessárias do PySide6
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# ==============================================================================
# FUNÇÕES DE DETECÇÃO REESCRITAS E OTIMIZADAS
# ==============================================================================

def detectar_desmatamento(imagem):
    """Detecta solo exposto (marrom/ocre)."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Faixa de cor para solo (marrom/ocre/areia)
    lower_solo = np.array([10, 40, 40])
    upper_solo = np.array([30, 255, 255])
    mascara_solo = cv2.inRange(hsv, lower_solo, upper_solo)
    
    kernel = np.ones((5,5), np.uint8)
    mascara_solo = cv2.morphologyEx(mascara_solo, cv2.MORPH_OPEN, kernel)
    mascara_solo = cv2.morphologyEx(mascara_solo, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(mascara_solo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    imagem_resultado = imagem.copy()
    area_total_desmatada = 0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(imagem_resultado, 'Desmatamento', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            area_total_desmatada += area
            
    # Função agora retorna apenas 2 valores, como no original
    return imagem_resultado, area_total_desmatada


def detectar_focos_incendio(imagem):
    """
    Detecta focos de incêndio com uma lógica mais robusta para brilho extremo.
    A nova regra é: um pixel é fogo se estiver na faixa de matiz (Hue) correta
    E tiver um brilho (Value) muito alto, independentemente da saturação.
    """
    blurred = cv2.GaussianBlur(imagem, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 1. Definir a faixa de MATIZ (HUE) para vermelho/laranja/amarelo
    # Faixa 1: para vermelhos (0-20)
    lower_hue1 = np.array([0, 0, 0])
    upper_hue1 = np.array([20, 255, 255])
    mask_hue1 = cv2.inRange(hsv, lower_hue1, upper_hue1)
    
    # Faixa 2: para vermelhos do outro lado do círculo de cores (170-180)
    lower_hue2 = np.array([170, 0, 0])
    upper_hue2 = np.array([180, 255, 255])
    mask_hue2 = cv2.inRange(hsv, lower_hue2, upper_hue2)

    mask_hue = cv2.bitwise_or(mask_hue1, mask_hue2)

    # 2. Definir a faixa de BRILHO (VALUE)
    # Apenas pixels com brilho acima de um limiar alto são considerados.
    # Este é o passo chave para detectar fogo "branco estourado".
    MIN_BRIGHTNESS = 215
    lower_val = np.array([0, 0, MIN_BRIGHTNESS])
    upper_val = np.array([180, 255, 255])
    mask_val = cv2.inRange(hsv, lower_val, upper_val)

    # 3. Combinar as máscaras
    # Um pixel deve satisfazer AMBAS as condições: ter a cor certa E ser muito brilhante.
    mascara_final = cv2.bitwise_and(mask_hue, mask_val)

    # 4. Limpeza morfológica
    kernel = np.ones((5,5), np.uint8)
    mascara_final = cv2.morphologyEx(mascara_final, cv2.MORPH_OPEN, kernel)
    mascara_final = cv2.morphologyEx(mascara_final, cv2.MORPH_DILATE, kernel) # Dilatar para juntar focos próximos

    # 5. Encontrar e filtrar contornos
    contornos, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_resultado = imagem.copy()
    focos_encontrados = 0
    MIN_AREA = 30 # Limiar de área pequeno para pegar focos menores

    for contorno in contornos:
        if cv2.contourArea(contorno) > MIN_AREA:
            focos_encontrados += 1
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(imagem_resultado, 'Foco de Incendio', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return imagem_resultado, focos_encontrados


# ==============================================================================
# CLASSE DA APLICAÇÃO (AJUSTADA PARA A LÓGICA MAIS SIMPLES)
# ==============================================================================
class DetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Desmatamento e Incêndio (PySide6)")
        self.setFixedSize(1200, 550)
        self.caminho_imagem = None
        
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50; color: #ecf0f1;
                font-family: 'Segoe UI'; font-size: 10pt;
            }
            QPushButton {
                background-color: #3498db; color: white; border: none;
                padding: 10px 15px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5dade2; }
            QPushButton:disabled { background-color: #95a5a6; color: #d5d8dc; }
            QLabel#ImageContainer {
                border: 2px solid #34495e; background-color: #34495e;
                border-radius: 5px;
            }
            QLabel { color: #ecf0f1; }
        """)
        self.layout_principal = QVBoxLayout(self)
        self.layout_controles = QHBoxLayout()
        self.layout_imagens = QHBoxLayout()
        self.btn_selecionar = QPushButton("Selecionar Imagem")
        self.lbl_caminho = QLabel("Nenhuma imagem selecionada")
        self.btn_processar = QPushButton("Processar Imagem")
        self.btn_processar.setEnabled(False)
        self.lbl_img_original = QLabel("Imagem Original")
        self.lbl_img_original.setAlignment(Qt.AlignCenter)
        self.lbl_img_original.setMinimumSize(400, 300)
        self.lbl_img_original.setObjectName("ImageContainer")
        self.lbl_resultado_desmatamento = QLabel()
        self.lbl_img_desmatamento = QLabel("Resultado Desmatamento")
        self.lbl_img_desmatamento.setAlignment(Qt.AlignCenter)
        self.lbl_img_desmatamento.setMinimumSize(400, 300)
        self.lbl_img_desmatamento.setObjectName("ImageContainer")
        self.lbl_resultado_incendio = QLabel()
        self.lbl_img_incendio = QLabel("Resultado Incêndio")
        self.lbl_img_incendio.setAlignment(Qt.AlignCenter)
        self.lbl_img_incendio.setMinimumSize(400, 300)
        self.lbl_img_incendio.setObjectName("ImageContainer")
        self.layout_controles.addWidget(self.btn_selecionar)
        self.layout_controles.addWidget(self.lbl_caminho, 1)
        self.layout_controles.addWidget(self.btn_processar)
        layout_original = QVBoxLayout()
        lbl_titulo_original = QLabel("<b>Imagem Original</b>")
        lbl_titulo_original.setAlignment(Qt.AlignCenter)
        layout_original.addWidget(lbl_titulo_original)
        layout_original.addWidget(self.lbl_img_original)
        layout_desmatamento = QVBoxLayout()
        self.lbl_resultado_desmatamento.setAlignment(Qt.AlignCenter)
        layout_desmatamento.addWidget(self.lbl_resultado_desmatamento)
        layout_desmatamento.addWidget(self.lbl_img_desmatamento)
        layout_incendio = QVBoxLayout()
        self.lbl_resultado_incendio.setAlignment(Qt.AlignCenter)
        layout_incendio.addWidget(self.lbl_resultado_incendio)
        layout_incendio.addWidget(self.lbl_img_incendio)
        self.layout_imagens.addLayout(layout_original)
        self.layout_imagens.addLayout(layout_desmatamento)
        self.layout_imagens.addLayout(layout_incendio)
        self.layout_principal.addLayout(self.layout_controles)
        self.layout_principal.addLayout(self.layout_imagens)
        self.btn_selecionar.clicked.connect(self.selecionar_e_exibir_imagem)
        self.btn_processar.clicked.connect(self.processar_imagem)

    def selecionar_e_exibir_imagem(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Arquivos de Imagem (*.png *.jpg *.jpeg *.bmp)")
        if caminho:
            self.caminho_imagem = caminho
            self.lbl_caminho.setText(caminho)
            self.btn_processar.setEnabled(True)
            self.limpar_resultados()
            imagem_original_cv = cv2.imread(self.caminho_imagem)
            if imagem_original_cv is not None:
                pixmap_original = self.converter_cv2_para_qpixmap(imagem_original_cv)
                self.lbl_img_original.setPixmap(pixmap_original.scaled(self.lbl_img_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                 QMessageBox.critical(self, "Erro", f"Não foi possível carregar a imagem em:\n{self.caminho_imagem}")

    def limpar_resultados(self):
        self.lbl_img_original.setText("Imagem Original")
        self.lbl_img_desmatamento.setText("Resultado Desmatamento")
        self.lbl_img_incendio.setText("Resultado Incêndio")
        self.lbl_resultado_desmatamento.clear()
        self.lbl_resultado_incendio.clear()

    def processar_imagem(self):
        if not self.caminho_imagem: return
        imagem_original = cv2.imread(self.caminho_imagem)
        if imagem_original is None:
            QMessageBox.critical(self, "Erro", f"Não foi possível carregar a imagem em:\n{self.caminho_imagem}")
            return

        # --- Processamento de Desmatamento ---
        resultado_desmatamento_cv, area_pixels = detectar_desmatamento(imagem_original.copy())
        self.lbl_resultado_desmatamento.setText(f"<b>Área de Desmatamento:</b> {int(area_pixels)} pixels")
        pixmap_desmatamento = self.converter_cv2_para_qpixmap(resultado_desmatamento_cv)
        self.lbl_img_desmatamento.setPixmap(pixmap_desmatamento.scaled(self.lbl_img_desmatamento.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # --- Processamento de Incêndio ---
        # A chamada foi simplificada, não precisa mais da máscara de solo
        resultado_incendio_cv, num_focos = detectar_focos_incendio(imagem_original.copy())
        self.lbl_resultado_incendio.setText(f"<b>Focos de Incêndio:</b> {num_focos}")
        pixmap_incendio = self.converter_cv2_para_qpixmap(resultado_incendio_cv)
        self.lbl_img_incendio.setPixmap(pixmap_incendio.scaled(self.lbl_img_incendio.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def converter_cv2_para_qpixmap(self, imagem_cv):
        imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = imagem_rgb.shape
        bytes_por_linha = ch * w
        q_img = QImage(imagem_rgb.data, w, h, bytes_por_linha, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectorApp()
    window.show()
    sys.exit(app.exec())
