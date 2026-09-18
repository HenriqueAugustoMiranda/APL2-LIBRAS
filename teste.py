# import cv2
# import mediapipe as mp

# print("oi")

# # Inicializar os módulos do MediaPipe
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# print("oi")
# # Iniciar captura de vídeo
# cap = cv2.VideoCapture(0)

# with mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as holistic:
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Converter para RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
        
#         # Processar o frame
#         results = holistic.process(image)

#         # Voltar para BGR para exibição
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # 1. Desenhar a malha do Rosto
#         if results.face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
#             )

#         # 2. Desenhar a Postura Corporal (Pose)
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
#             )

#         # 3. Desenhar Mão Esquerda e Direita
#         if results.left_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
#             )
#         if results.right_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
#             )

#         cv2.imshow('Reconhecedor de LIBRAS - Captura de Pontos', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Desenha a malha do rosto, mãos e corpo se detectados
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Teste LIBRAS - MediaPipe', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()