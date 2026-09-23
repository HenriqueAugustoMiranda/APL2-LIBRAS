import cv2
import mediapipe as mp
import math
import numpy as np

def calcular_distancia(p1, p2, largura_img, altura_img):
    x1, y1 = int(p1.x * largura_img), int(p1.y * altura_img)
    x2, y2 = int(p2.x * largura_img), int(p2.y * altura_img)
    return math.hypot(x2 - x1, y2 - y1)

# --- 1. EAR (Eye Aspect Ratio) ---
def calcular_ear(olho_landmarks, landmarks, w, h):
    v1 = calcular_distancia(landmarks[olho_landmarks[0]], landmarks[olho_landmarks[1]], w, h)
    v2 = calcular_distancia(landmarks[olho_landmarks[2]], landmarks[olho_landmarks[3]], w, h)
    h_dist = calcular_distancia(landmarks[olho_landmarks[4]], landmarks[olho_landmarks[5]], w, h)
    
    if h_dist == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)

# --- 2. MAR (Mouth Aspect Ratio - Abertura Vertical) ---
def calcular_mar(landmarks, w, h):
    v1 = calcular_distancia(landmarks[13], landmarks[14], w, h)      # Lábio sup/inf centro
    v2 = calcular_distancia(landmarks[82], landmarks[87], w, h)      # Lábio sup/inf esq
    v3 = calcular_distancia(landmarks[312], landmarks[317], w, h)    # Lábio sup/inf dir
    h_dist = calcular_distancia(landmarks[61], landmarks[291], w, h)  # Cantos da boca

    if h_dist == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h_dist)

# --- 3. BAR (Brow Aspect Ratio - Sobrancelhas) ---
def calcular_bar(landmarks, w, h):
    dist_sob_esq = calcular_distancia(landmarks[70], landmarks[159], w, h)
    dist_sob_dir = calcular_distancia(landmarks[300], landmarks[386], w, h)
    dist_interocular = calcular_distancia(landmarks[33], landmarks[263], w, h)
    dist_entre_sobrancelhas = calcular_distancia(landmarks[70], landmarks[300], w, h)

    if dist_interocular == 0:
        return 0.0, 0.0

    altura_relativa = ((dist_sob_esq + dist_sob_dir) / 2.0) / dist_interocular
    dist_relativa_juntas = dist_entre_sobrancelhas / dist_interocular
    
    return altura_relativa, dist_relativa_juntas

# --- 4. NMAR (Normalized Mouth Corner Ratio - Sorriso e Esticamento) ---
def calcular_nmar(landmarks, w, h):
    dist_boca_largura = calcular_distancia(landmarks[61], landmarks[291], w, h)
    dist_interocular = calcular_distancia(landmarks[33], landmarks[263], w, h)
    
    if dist_interocular == 0:
        return 0.0
    return dist_boca_largura / dist_interocular

# --- 5. PUP (Pucker Ratio - Bico / Lábios Projetados) ---
def calcular_pup(landmarks, w, h):
    dist_labio_externo_altura = calcular_distancia(landmarks[0], landmarks[17], w, h)
    dist_boca_largura = calcular_distancia(landmarks[61], landmarks[291], w, h)
    
    if dist_boca_largura == 0:
        return 0.0
    return dist_labio_externo_altura / dist_boca_largura

# --- 6. CAR (Cheek Aspect Ratio - Bochechas Infladas) ---
def calcular_car(landmarks, w, h):
    dist_bochechas = calcular_distancia(landmarks[234], landmarks[454], w, h)
    dist_nariz_queixo = calcular_distancia(landmarks[1], landmarks[152], w, h)
    
    if dist_nariz_queixo == 0:
        return 0.0
    return dist_bochechas / dist_nariz_queixo

# --- 7. NAR (Nasal Aspect Ratio - Nariz Franzido) ---
def calcular_nar(landmarks, w, h):
    dist_dorso_nasal = calcular_distancia(landmarks[198], landmarks[2], w, h)
    dist_interna_olhos = calcular_distancia(landmarks[133], landmarks[362], w, h)
    
    if dist_interna_olhos == 0:
        return 0.0
    return dist_dorso_nasal / dist_interna_olhos

# --- FUNÇÃO AUXILIAR SOFTMAX ---
def calcular_probabilidades(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

PONTOS_OLHO_ESQ = [160, 144, 158, 153, 33, 133]
PONTOS_OLHO_DIR = [385, 380, 387, 373, 362, 263]

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera ou fim do vídeo.")
            break

        altura, largura, _ = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark

            # 1. Desenhar a malha do rosto
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )

            # 2. CÁLCULO DE TODOS OS ASPECT RATIOS
            ear_esq = calcular_ear(PONTOS_OLHO_ESQ, landmarks, largura, altura)
            ear_dir = calcular_ear(PONTOS_OLHO_DIR, landmarks, largura, altura)
            ear_medio = (ear_esq + ear_dir) / 2.0

            mar = calcular_mar(landmarks, largura, altura)
            bar_altura, bar_juntas = calcular_bar(landmarks, largura, altura)
            nmar = calcular_nmar(landmarks, largura, altura)
            pup = calcular_pup(landmarks, largura, altura)
            car = calcular_car(landmarks, largura, altura)
            nar = calcular_nar(landmarks, largura, altura)

            # 3. CÁLCULO DE SCORES CONTÍNUOS
            score_pergunta_geral = ((bar_altura - 0.25) / 0.10) + ((ear_medio - 0.22) / 0.08)
            score_pergunta_especifica = ((0.35 - bar_juntas) / 0.08) + ((0.42 - nar) / 0.08)
            score_bico = ((pup - 0.35) / 0.12) - ((mar - 0.15) / 0.10)
            score_bochechas = (car - 1.45) / 0.20
            score_boca_aberta = (mar - 0.25) / 0.15
            score_sorriso = ((nmar - 0.60) / 0.15) - ((mar - 0.15) / 0.10)
            score_neutro = 1.0 - (abs(bar_altura - 0.26) + abs(mar - 0.15) + abs(nmar - 0.65))

            scores = np.array([
                score_pergunta_geral,
                score_pergunta_especifica,
                score_bico,
                score_bochechas,
                score_boca_aberta,
                score_sorriso,
                score_neutro
            ])

            nomes_expressoes = [
                "Pergunta Geral (Sim/Nao)",
                "Pergunta Especifica (Quem/Onde)",
                "Bico ('PSI/PE')",
                "Bochechas Infladas",
                "Boca Aberta / Enfase",
                "Sorriso / Confirmacao",
                "Neutro"
            ]

            # Converter os scores em probabilidades (0.0 a 1.0)
            probabilidades = calcular_probabilidades(scores)

            # Ordenar da maior chance para a menor
            chances_ordenadas = sorted(
                zip(nomes_expressoes, probabilidades),
                key=lambda item: item[1],
                reverse=True
            )

            # Exibir as 3 maiores probabilidades na tela
            y_offset = 30
            cv2.putText(image, "Probabilidades de Expressao:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            for i in range(3):
                nome, prob = chances_ordenadas[i]
                y_offset += 25
                cor = (0, 255, 0) if i == 0 else (200, 200, 200)  # Destaque verde no 1º lugar
                texto = f"{i+1}. {nome}: {prob*100:.1f}%"
                cv2.putText(image, texto, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA)

            # Telemetria na tela para depuração
            info_texto = f"BAR:{bar_altura:.2f} | MAR:{mar:.2f} | NMAR:{nmar:.2f} | PUP:{pup:.2f} | CAR:{car:.2f} | NAR:{nar:.2f}"
            cv2.putText(image, info_texto, (10, altura - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # --- DESENHO DOS OUTROS PONTOS (Mãos e Corpo) ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Reconhecedor de LIBRAS - Captura de Pontos', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()