import cv2
import mediapipe as mp
import math


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b (in degrees) given three points a, b, c.
    Each point is a tuple (x, y).
    """
    BA = (a[0] - b[0], a[1] - b[1])
    BC = (c[0] - b[0], c[1] - b[1])
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    magBA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    magBC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    if magBA * magBC == 0:
        return 0.0
    cos_angle = dot_product / (magBA * magBC)
    # Clamp due to floating point errors.
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


def is_hand_closed(hand_landmarks, w, h):
    """
    Determine if the hand is closed using a relative measure.
    We compare the distance between the thumb tip and index tip to the distance between
    the index MCP and index tip. If the ratio is less than 0.8, we consider the hand closed.
    """
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    index_mcp = hand_landmarks[5]

    thumb_tip_xy = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_tip_xy = (int(index_tip.x * w), int(index_tip.y * h))
    index_mcp_xy = (int(index_mcp.x * w), int(index_mcp.y * h))

    # Compute Euclidean distances.
    dist_thumb_index = math.sqrt((thumb_tip_xy[0] - index_tip_xy[0]) ** 2 +
                                 (thumb_tip_xy[1] - index_tip_xy[1]) ** 2)
    dist_index = math.sqrt((index_tip_xy[0] - index_mcp_xy[0]) ** 2 +
                           (index_tip_xy[1] - index_mcp_xy[1]) ** 2)
    if dist_index == 0:
        return False
    ratio = dist_thumb_index / dist_index
    return ratio < 0.8


# Initialize MediaPipe.
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Open your HD camera and set resolution to HD.
cap = cv2.VideoCapture(1)

# Initialize variables for left and right arms/hands.
left_shoulder_angle = 0.0
left_elbow_angle = 0.0
left_wrist_angle = 0.0
left_finger_angle = 0.0
left_claw_state = "UNKNOWN"
left_yaw = 0.0

right_shoulder_angle = 0.0
right_elbow_angle = 0.0
right_wrist_angle = 0.0
right_finger_angle = 0.0
right_claw_state = "UNKNOWN"
right_yaw = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process with MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        h, w, _ = frame.shape

        # ---------------- Process Pose Landmarks (Arms) ----------------
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # ----- Left Arm -----
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            left_elbow = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
            left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            # Compute left yaw: angle from shoulder to wrist relative to horizontal.
            left_yaw = math.degrees(math.atan2(left_wrist[1] - left_shoulder[1],
                                               left_wrist[0] - left_shoulder[0]))
            if left_yaw < 0:
                left_yaw += 360

            # ----- Right Arm -----
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            right_elbow = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            right_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))
            right_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            # Compute right yaw.
            right_yaw = math.degrees(math.atan2(right_wrist[1] - right_shoulder[1],
                                                right_wrist[0] - right_shoulder[0]))
            if right_yaw < 0:
                right_yaw += 360

            # Determine "visibility score" for each arm (average of shoulder, elbow, wrist visibilities).
            left_score = (landmarks[11].visibility + landmarks[13].visibility + landmarks[15].visibility) / 3.0
            right_score = (landmarks[12].visibility + landmarks[14].visibility + landmarks[16].visibility) / 3.0
            # Choose the main arm.
            if left_score >= right_score:
                main_arm = 'left'
            else:
                main_arm = 'right'
        # ----------------------------------------------------------------

        # ---------------- Process Hand Landmarks ----------------
        # Process left hand.
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks.landmark
            left_index_mcp = (int(left_hand[5].x * w), int(left_hand[5].y * h))
            # Update left wrist angle using hand data.
            left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index_mcp)
            # Finger angle using index finger landmarks (MCP, PIP, TIP).
            left_index_pip = (int(left_hand[6].x * w), int(left_hand[6].y * h))
            left_index_tip = (int(left_hand[8].x * w), int(left_hand[8].y * h))
            left_finger_angle = calculate_angle(left_index_mcp, left_index_pip, left_index_tip)
            left_claw_state = "CLOSED" if is_hand_closed(left_hand, w, h) else "OPEN"

        # Process right hand.
        if results.right_hand_landmarks:
            right_hand = results.right_hand_landmarks.landmark
            right_index_mcp = (int(right_hand[5].x * w), int(right_hand[5].y * h))
            right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_index_mcp)
            right_index_pip = (int(right_hand[6].x * w), int(right_hand[6].y * h))
            right_index_tip = (int(right_hand[8].x * w), int(right_hand[8].y * h))
            right_finger_angle = calculate_angle(right_index_mcp, right_index_pip, right_index_tip)
            right_claw_state = "CLOSED" if is_hand_closed(right_hand, w, h) else "OPEN"
        # ----------------------------------------------------------

        # ---------------- Visualization ----------------
        # Draw only arms and hands.
        if results.pose_landmarks:
            def to_pixel(idx):
                return (int(results.pose_landmarks.landmark[idx].x * w),
                        int(results.pose_landmarks.landmark[idx].y * h))


            # Determine colors: main arm is green, the other is red.
            left_color = (0, 255, 0) if (results.pose_landmarks.landmark[11].visibility +
                                         results.pose_landmarks.landmark[13].visibility +
                                         results.pose_landmarks.landmark[15].visibility) / 3.0 >= \
                                        (results.pose_landmarks.landmark[12].visibility +
                                         results.pose_landmarks.landmark[14].visibility +
                                         results.pose_landmarks.landmark[16].visibility) / 3.0 else (0, 0, 255)
            right_color = (0, 255, 0) if left_color == (0, 0, 255) else (0, 0, 255)
            # Draw left arm.
            cv2.line(frame, to_pixel(11), to_pixel(13), left_color, 2)
            cv2.line(frame, to_pixel(13), to_pixel(15), left_color, 2)
            # Draw right arm.
            cv2.line(frame, to_pixel(12), to_pixel(14), right_color, 2)
            cv2.line(frame, to_pixel(14), to_pixel(16), right_color, 2)

            # Display yaw for the main arm.
            if main_arm == 'left':
                cv2.putText(frame, f"Yaw: {int(left_yaw)}", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
            else:
                cv2.putText(frame, f"Yaw: {int(right_yaw)}", (650, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)

        # Draw hand landmarks.
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # -------------------------------------------------

        # Display computed angles and claw state.
        cv2.putText(frame, f"L-Shoulder: {int(left_shoulder_angle)}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"L-Elbow: {int(left_elbow_angle)}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"L-Wrist: {int(left_wrist_angle)}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"L-Finger: {int(left_finger_angle)}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"L-Claw: {left_claw_state}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"R-Shoulder: {int(right_shoulder_angle)}", (650, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"R-Elbow: {int(right_elbow_angle)}", (650, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"R-Wrist: {int(right_wrist_angle)}", (650, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"R-Finger: {int(right_finger_angle)}", (650, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"R-Claw: {right_claw_state}", (650, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Arm + Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()