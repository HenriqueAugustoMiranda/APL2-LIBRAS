import cv2
import mediapipe as mp
import math
import numpy as np

# --- DISTÂNCIA EUCLIDIANA 2D E 3D ---
def calcular_distancia_2d(p1, p2, largura_img, altura_img):
    x1, y1 = int(p1.x * largura_img), int(p1.y * altura_img)
    x2, y2 = int(p2.x * largura_img), int(p2.y * altura_img)
    return math.hypot(x2 - x1, y2 - y1)

def calcular_distancia_3d(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

# --- ESTIMATIVA DE ANGULAÇÃO FACIAL (YAW - ROTAÇÃO HORIZONTAL) ---
def estimar_yaw(landmarks):
    # Proporção entre a ponta do nariz (1) e as orelhas/bochechas (234, 454)
    dist_esq = abs(landmarks[1].x - landmarks[234].x)
    dist_dir = abs(landmarks[454].x - landmarks[1].x)
    total = dist_esq + dist_dir
    if total == 0:
        return 0.0
    # Retorna valor relativo: ~0.0 de frente, negativo olhado para a esquerda, positivo para a direita
    return (dist_dir - dist_esq) / total

# --- 1. EAR (Eye Aspect Ratio) ---
def calcular_ear(olho_landmarks, landmarks, w, h):
    v1 = calcular_distancia_2d(landmarks[olho_landmarks[0]], landmarks[olho_landmarks[1]], w, h)
    v2 = calcular_distancia_2d(landmarks[olho_landmarks[2]], landmarks[olho_landmarks[3]], w, h)
    h_dist = calcular_distancia_2d(landmarks[olho_landmarks[4]], landmarks[olho_landmarks[5]], w, h)
    if h_dist == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)

# --- 2. MAR (Mouth Aspect Ratio) ---
def calcular_mar(landmarks, w, h):
    v1 = calcular_distancia_2d(landmarks[13], landmarks[14], w, h)
    v2 = calcular_distancia_2d(landmarks[82], landmarks[87], w, h)
    v3 = calcular_distancia_2d(landmarks[312], landmarks[317], w, h)
    h_dist = calcular_distancia_2d(landmarks[61], landmarks[291], w, h)
    if h_dist == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h_dist)

# --- 3. BAR (Brow Aspect Ratio - Usando 3D para compensação de perspectiva) ---
def calcular_bar(landmarks, w, h):
    dist_sob_esq = calcular_distancia_2d(landmarks[70], landmarks[159], w, h)
    dist_sob_dir = calcular_distancia_2d(landmarks[300], landmarks[386], w, h)
    dist_interocular = calcular_distancia_2d(landmarks[33], landmarks[263], w, h)

    if dist_interocular == 0:
        return 0.0, 0.0

    altura_relativa = ((dist_sob_esq + dist_sob_dir) / 2.0) / dist_interocular

    # Uso de coordenadas 3D (x, y, z) do MediaPipe para franzimento imune à rotação da cabeça
    dist_horiz_3d = calcular_distancia_3d(landmarks[55], landmarks[285])
    dist_vert_esq_3d = calcular_distancia_3d(landmarks[55], landmarks[9])
    dist_vert_dir_3d = calcular_distancia_3d(landmarks[285], landmarks[9])
    dist_interocular_3d = calcular_distancia_3d(landmarks[33], landmarks[263])

    if dist_interocular_3d == 0:
        return altura_relativa, 0.0

    fator_franzimento = (dist_horiz_3d + dist_vert_esq_3d + dist_vert_dir_3d) / 3.0
    dist_relativa_juntas = fator_franzimento / dist_interocular_3d

    return altura_relativa, dist_relativa_juntas

# --- 4. NMAR ---
def calcular_nmar(landmarks, w, h):
    dist_boca_largura = calcular_distancia_2d(landmarks[61], landmarks[291], w, h)
    dist_interocular = calcular_distancia(landmarks[33], landmarks[263], w, h) if 'calcular_distancia' in globals() else calcular_distancia_2d(landmarks[33], landmarks[263], w, h)
    if dist_interocular == 0:
        return 0.0
    return dist_boca_largura / dist_interocular

# --- 5. PUP ---
def calcular_pup(landmarks, w, h):
    dist_labio_externo_altura = calcular_distancia_2d(landmarks[0], landmarks[17], w, h)
    dist_boca_largura = calcular_distancia_2d(landmarks[61], landmarks[291], w, h)
    if dist_boca_largura == 0:
        return 0.0
    return dist_labio_externo_altura / dist_boca_largura

# --- 6. CAR ---
def calcular_car(landmarks, w, h):
    dist_bochechas = calcular_distancia_2d(landmarks[234], landmarks[454], w, h)
    dist_nariz_queixo = calcular_distancia_2d(landmarks[1], landmarks[152], w, h)
    if dist_nariz_queixo == 0:
        return 0.0
    return dist_bochechas / dist_nariz_queixo

# --- 7. NAR ---
def calcular_nar(landmarks, w, h):
    dist_dorso_nasal = calcular_distancia_2d(landmarks[198], landmarks[2], w, h)
    dist_interna_olhos = calcular_distancia_2d(landmarks[133], landmarks[362], w, h)
    if dist_interna_olhos == 0:
        return 0.0
    return dist_dorso_nasal / dist_interna_olhos

# --- SOFTMAX ---
def calcular_probabilidades(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

PONTOS_OLHO_ESQ = [160, 144, 158, 153, 33, 133]
PONTOS_OLHO_DIR = [385, 380, 387, 373, 362, 263]
PONTOS_BAR = [70, 159, 300, 386, 33, 263, 55, 285, 9]

cap = cv2.VideoCapture(0)

NOME_JANELA = 'Reconhecedor de LIBRAS - Mapeamento e Calibração'
cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOME_JANELA, 800, 600)

# --- ESTADOS DO MAPEAMENTO/CALIBRAÇÃO ---
# Estágios: 'FRENTE', 'DIREITA', 'ESQUERDA', 'CONCLUIDO'
estagio_calibracao = 'FRENTE'
frames_estagio = 0
FRAMES_NECESSARIOS = 30  # Captura ~1 segundo em cada posição

# Armazenamento de vetores neutros por perfil
calib_frente = []
calib_direita = []
calib_esquerda = []

perfis_neutros = {
    'frente': None,
    'direita': None,
    'esquerda': None
}

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        altura, largura, _ = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # SE O ROSTO SUMIR DA CÂMERA -> RESET COMPLETO DA PESSOA
        if not results.face_landmarks:
            estagio_calibracao = 'FRENTE'
            frames_estagio = 0
            calib_frente.clear()
            calib_direita.clear()
            calib_esquerda.clear()
            perfis_neutros = {'frente': None, 'direita': None, 'esquerda': None}

            cv2.putText(image, "Aguardando rosto para iniciar calibracao...", (30, altura // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            landmarks = results.face_landmarks.landmark
            yaw = estimar_yaw(landmarks)

            # 1. Desenhar malha do rosto
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )

            # CÁLCULO DOS RATIOS
            ear_esq = calcular_ear(PONTOS_OLHO_ESQ, landmarks, largura, altura)
            ear_dir = calcular_ear(PONTOS_OLHO_DIR, landmarks, largura, altura)
            ear_medio = (ear_esq + ear_dir) / 2.0

            mar = calcular_mar(landmarks, largura, altura)
            bar_altura, bar_juntas = calcular_bar(landmarks, largura, altura)
            nmar = calcular_nmar(landmarks, largura, altura)
            pup = calcular_pup(landmarks, largura, altura)
            car = calcular_car(landmarks, largura, altura)
            nar = calcular_nar(landmarks, largura, altura)

            vetor_atual = np.array([bar_altura, bar_juntas, ear_medio, mar, nmar, pup, car, nar])

            # --- ETAPA DE MAPEAMENTO / CALIBRAÇÃO PASSO A PASSO ---
            if estagio_calibracao != 'CONCLUIDO':
                cv2.rectangle(image, (20, 20), (largura - 20, 120), (0, 0, 0), -1)

                if estagio_calibracao == 'FRENTE':
                    instrucao = f"Fique de FRENTE com expressao NEUTRA ({frames_estagio}/{FRAMES_NECESSARIOS})"
                    if abs(yaw) < 0.12:  # Garante que a pessoa está realmente de frente
                        calib_frente.append(vetor_atual)
                        frames_estagio += 1
                        if frames_estagio >= FRAMES_NECESSARIOS:
                            perfis_neutros['frente'] = np.mean(calib_frente, axis=0)
                            estagio_calibracao = 'DIREITA'
                            frames_estagio = 0
                    else:
                        instrucao += " [Olhe diretamente para a camera!]"

                elif estagio_calibracao == 'DIREITA':
                    instrucao = f"Vire um pouco para a DIREITA (neutro) ({frames_estagio}/{FRAMES_NECESSARIOS})"
                    if yaw > 0.12:  # Garante que a cabeça virou para a direita
                        calib_direita.append(vetor_atual)
                        frames_estagio += 1
                        if frames_estagio >= FRAMES_NECESSARIOS:
                            perfis_neutros['direita'] = np.mean(calib_direita, axis=0)
                            estagio_calibracao = 'ESQUERDA'
                            frames_estagio = 0
                    else:
                        instrucao += " [Gire a cabeca mais para a direita!]"

                elif estagio_calibracao == 'ESQUERDA':
                    instrucao = f"Vire um pouco para a ESQUERDA (neutro) ({frames_estagio}/{FRAMES_NECESSARIOS})"
                    if yaw < -0.12:  # Garante que a cabeça virou para a esquerda
                        calib_esquerda.append(vetor_atual)
                        frames_estagio += 1
                        if frames_estagio >= FRAMES_NECESSARIOS:
                            perfis_neutros['esquerda'] = np.mean(calib_esquerda, axis=0)
                            estagio_calibracao = 'CONCLUIDO'

                cv2.putText(image, "CALIBRANDO ROSTO", (30, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, instrucao, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- ETAPA DE RECONHECIMENTO (PÓS-CALIBRAÇÃO) ---
            else:
                # Seleção dinâmica do neutro com base na angulação da cabeça
                if yaw > 0.10:
                    neutro_ref = perfis_neutros['direita']
                    posicao_cabeca = "Direita"
                elif yaw < -0.10:
                    neutro_ref = perfis_neutros['esquerda']
                    posicao_cabeca = "Esquerda"
                else:
                    neutro_ref = perfis_neutros['frente']
                    posicao_cabeca = "Frente"

                # Desempacotar vetor neutro dinâmico
                b_alt_n, b_juntas_n, ear_n, mar_n, nmar_n, pup_n, car_n, nar_n = neutro_ref

                # Variação relativa ao neutro calibrado da angulação
                queda_juntas = (b_juntas_n - bar_juntas) / b_juntas_n if b_juntas_n > 0 else 0

                # SCORES CONTÍNUOS
                score_pergunta_geral = ((bar_altura - b_alt_n) / 0.08) + ((ear_medio - ear_n) / 0.06)
                score_pergunta_especifica = ((queda_juntas - 0.08) / 0.10) + ((nar_n - nar) / 0.06)
                score_bico = ((pup - pup_n) / 0.10) - ((mar - mar_n) / 0.10)
                score_bochechas = (car - car_n) / 0.15
                score_boca_aberta = (mar - mar_n) / 0.12
                score_sorriso = ((nmar - nmar_n) / 0.10) - ((mar - mar_n) / 0.10)
                
                # Neutro: quão próximo o vetor atual está da sua referência de ângulo
                dist_do_neutro = np.linalg.norm(vetor_atual - neutro_ref)
                score_neutro = 1.0 - (dist_do_neutro * 2.0)

                scores = np.array([
                    score_pergunta_geral, score_pergunta_especifica, score_bico,
                    score_bochechas, score_boca_aberta, score_sorriso, score_neutro
                ])

                nomes_expressoes = [
                    "Pergunta Geral (Sim/Nao)", "Pergunta Especifica (Quem/Onde)",
                    "Bico ('PSI/PE')", "Bochechas Infladas", "Boca Aberta / Enfase",
                    "Sorriso / Confirmacao", "Neutro"
                ]

                probabilidades = calcular_probabilidades(scores)
                chances_ordenadas = sorted(zip(nomes_expressoes, probabilidades), key=lambda item: item[1], reverse=True)

                # RENDERIZAÇÃO NA TELA
                y_offset = 30
                cv2.putText(image, f"Mapeado OK | Orientacao: {posicao_cabeca}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                for i in range(3):
                    nome, prob = chances_ordenadas[i]
                    y_offset += 25
                    cor = (0, 255, 0) if i == 0 else (200, 200, 200)
                    cv2.putText(image, f"{i+1}. {nome}: {prob*100:.1f}%", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA)

                info_texto = f"BAR_alt:{bar_altura:.2f} | MAR:{mar:.2f} | Yaw:{yaw:.2f}"
                cv2.putText(image, info_texto, (10, altura - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Desenho Pose e Mãos
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow(NOME_JANELA, image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()