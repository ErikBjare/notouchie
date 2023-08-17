import cv2
import mediapipe as mp


def is_touching_face(hand_landmarks, face_bbox):
    for landmark in hand_landmarks.landmark:
        if (
            face_bbox.xmin <= landmark.x <= face_bbox.xmin + face_bbox.width
            and face_bbox.ymin <= landmark.y <= face_bbox.ymin + face_bbox.height
        ):
            return True
    return False


def main():
    # Initialize mediapipe hands and face components
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    hands = mp_hands.Hands()
    face = mp_face.FaceDetection(min_detection_confidence=0.2)

    # Initialize mediapipe drawing utility
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        face_results = face.process(rgb_frame)

        # Draw the hand landmarks
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Check if hand is touching face
        if hand_results.multi_hand_landmarks and face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                for landmarks in hand_results.multi_hand_landmarks:
                    if is_touching_face(landmarks, bboxC):
                        cv2.putText(
                            frame,
                            "TOUCHING FACE!",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
