import cv2
import mediapipe as mp
import math
import numpy as np
import time

def calculate_angle(a, b, c):
    BA = (a[0] - b[0], a[1] - b[1])
    BC = (c[0] - b[0], c[1] - b[1])
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    magBA = math.sqrt(BA[0]**2 + BA[1]**2)
    magBC = math.sqrt(BC[0]**2 + BC[1]**2)
    if magBA * magBC == 0:
        return 0.0
    cos_angle = dot_product / (magBA * magBC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def is_hand_closed(hand_landmarks, w, h):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    index_mcp = hand_landmarks[5]

    thumb_tip_xy = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_tip_xy = (int(index_tip.x * w), int(index_tip.y * h))
    index_mcp_xy = (int(index_mcp.x * w), int(index_mcp.y * h))

    dist_thumb_index = math.sqrt((thumb_tip_xy[0] - index_tip_xy[0])**2 +
                                 (thumb_tip_xy[1] - index_tip_xy[1])**2)
    dist_index = math.sqrt((index_tip_xy[0] - index_mcp_xy[0])**2 +
                           (index_tip_xy[1] - index_mcp_xy[1])**2)
    if dist_index == 0:
        return False
    ratio = dist_thumb_index / dist_index
    return ratio < 0.8

def clamp_change(current, target, max_delta):
    delta = target - current
    if abs(delta) > max_delta:
        return current + math.copysign(max_delta, delta)
    else:
        return target

# ---------------------------------------------------------------------
# Original Setup
# ---------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(1)

left_shoulder_angle = 0.0
left_elbow_angle   = 0.0
left_wrist_angle   = 0.0
left_claw_state    = "UNKNOWN"
left_yaw           = 0.0
left_wrist_pitch   = 0.0

right_shoulder_angle = 0.0
right_elbow_angle    = 0.0
right_wrist_angle    = 0.0
right_claw_state     = "UNKNOWN"
right_yaw            = 0.0
right_wrist_pitch    = 0.0

smoothed_motor6 = None  # Base rotation (top-down)
smoothed_motor5 = None  # Shoulder pitch
smoothed_motor4 = None  # Elbow
smoothed_motor3 = None  # Wrist pitch

max_speed = 72  # deg/s
last_time = time.time()

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        max_delta = max_speed * dt

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        h, w, _ = frame.shape

        # Pose Landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Left
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            left_elbow    = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            left_wrist    = (int(landmarks[15].x * w), int(landmarks[15].y * h))
            left_hip      = (int(landmarks[23].x * w), int(landmarks[23].y * h))

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

            left_yaw = math.degrees(math.atan2(left_wrist[1] - left_shoulder[1],
                                               left_wrist[0] - left_shoulder[0]))
            if left_yaw < 0:
                left_yaw += 360

            # Right
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            right_elbow    = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            right_wrist    = (int(landmarks[16].x * w), int(landmarks[16].y * h))
            right_hip      = (int(landmarks[24].x * w), int(landmarks[24].y * h))

            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

            right_yaw = math.degrees(math.atan2(right_wrist[1] - right_shoulder[1],
                                                right_wrist[0] - right_shoulder[0]))
            if right_yaw < 0:
                right_yaw += 360

            # Decide main arm
            left_score = (landmarks[11].visibility + landmarks[13].visibility + landmarks[15].visibility) / 3.0
            right_score= (landmarks[12].visibility + landmarks[14].visibility + landmarks[16].visibility) / 3.0
            main_arm = 'left' if left_score >= right_score else 'right'

        # Hand Landmarks
        left_hand_yaw = 0
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks.landmark
            thumb_tip = (int(left_hand[4].x * w), int(left_hand[4].y * h))
            index_tip = (int(left_hand[8].x * w), int(left_hand[8].y * h))

            left_hand_yaw = math.degrees(math.atan2(index_tip[1] - thumb_tip[1],
                                                     index_tip[0] - thumb_tip[0]))
            if left_hand_yaw < 0:
                left_hand_yaw += 360

            if len(left_hand) >= 13:
                left_middle_tip = (int(left_hand[12].x * w), int(left_hand[12].y * h))
                forearm_vec = (left_wrist[0] - left_elbow[0], left_wrist[1] - left_elbow[1])
                hand_vec   = (left_middle_tip[0] - left_wrist[0], left_middle_tip[1] - left_wrist[1])
                norm_forearm = math.sqrt(forearm_vec[0]**2 + forearm_vec[1]**2)
                norm_hand    = math.sqrt(hand_vec[0]**2 + hand_vec[1]**2)
                if norm_forearm * norm_hand != 0:
                    cos_angle = (forearm_vec[0]*hand_vec[0] + forearm_vec[1]*hand_vec[1]) / (norm_forearm*norm_hand)
                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                    left_wrist_pitch = math.degrees(math.acos(cos_angle))
                else:
                    left_wrist_pitch = 0
            else:
                left_wrist_pitch = 0

            left_claw_state = "CLOSED" if is_hand_closed(left_hand, w, h) else "OPEN"

        right_hand_yaw = 0
        if results.right_hand_landmarks:
            right_hand = results.right_hand_landmarks.landmark
            thumb_tip = (int(right_hand[4].x * w), int(right_hand[4].y * h))
            index_tip = (int(right_hand[8].x * w), int(right_hand[8].y * h))

            right_hand_yaw = math.degrees(math.atan2(index_tip[1] - thumb_tip[1],
                                                     index_tip[0] - thumb_tip[0]))
            if right_hand_yaw < 0:
                right_hand_yaw += 360

            if len(right_hand) >= 13:
                right_middle_tip = (int(right_hand[12].x * w), int(right_hand[12].y * h))
                forearm_vec = (right_wrist[0] - right_elbow[0], right_wrist[1] - right_elbow[1])
                hand_vec   = (right_middle_tip[0] - right_wrist[0], right_middle_tip[1] - right_wrist[1])
                norm_forearm = math.sqrt(forearm_vec[0]**2 + forearm_vec[1]**2)
                norm_hand    = math.sqrt(hand_vec[0]**2 + hand_vec[1]**2)
                if norm_forearm*norm_hand != 0:
                    cos_angle = (forearm_vec[0]*hand_vec[0] + forearm_vec[1]*hand_vec[1]) / (norm_forearm*norm_hand)
                    cos_angle = max(min(cos_angle, 1.0), -1.0)
                    right_wrist_pitch = math.degrees(math.acos(cos_angle))
                else:
                    right_wrist_pitch = 0
            else:
                right_wrist_pitch = 0

            right_claw_state = "CLOSED" if is_hand_closed(right_hand, w, h) else "OPEN"


        # Choose which side
        if main_arm == 'left':
            base_yaw          = left_yaw
            shoulder_angle_sim= left_shoulder_angle
            elbow_angle_sim   = left_elbow_angle
            wrist_angle_sim   = left_wrist_pitch
            hand_yaw_sim      = left_hand_yaw
            claw_state_sim    = left_claw_state

        else:
            base_yaw          = right_yaw
            shoulder_angle_sim= right_shoulder_angle
            elbow_angle_sim   = right_elbow_angle
            wrist_angle_sim   = right_wrist_pitch
            hand_yaw_sim      = right_hand_yaw
            claw_state_sim    = right_claw_state


        # Motors
        motor6 = base_yaw
        motor5 = max(0, shoulder_angle_sim - 90)
        motor4 = 180 - elbow_angle_sim
        motor3 = wrist_angle_sim

        # Rate-limit
        if smoothed_motor6 is None:
            smoothed_motor6 = motor6
        else:
            smoothed_motor6 = clamp_change(smoothed_motor6, motor6, max_delta)

        if smoothed_motor5 is None:
            smoothed_motor5 = motor5
        else:
            smoothed_motor5 = clamp_change(smoothed_motor5, motor5, max_delta)

        if smoothed_motor4 is None:
            smoothed_motor4 = motor4
        else:
            smoothed_motor4 = clamp_change(smoothed_motor4, motor4, max_delta)

        if smoothed_motor3 is None:
            smoothed_motor3 = motor3
        else:
            smoothed_motor3 = clamp_change(smoothed_motor3, motor3, max_delta)


        # Skeleton
        if results.pose_landmarks:
            def to_pixel(idx):
                return (int(results.pose_landmarks.landmark[idx].x * w),
                        int(results.pose_landmarks.landmark[idx].y * h))
            left_score = (landmarks[11].visibility + landmarks[13].visibility + landmarks[15].visibility) / 3.0
            right_score= (landmarks[12].visibility + landmarks[14].visibility + landmarks[16].visibility) / 3.0

            left_color  = (0, 255, 0) if left_score >= right_score else (0, 0, 255)
            right_color = (0, 255, 0) if left_score <  right_score else (0, 0, 255)

            cv2.line(frame, to_pixel(11), to_pixel(13), left_color, 2)
            cv2.line(frame, to_pixel(13), to_pixel(15), left_color, 2)
            cv2.line(frame, to_pixel(12), to_pixel(14), right_color, 2)
            cv2.line(frame, to_pixel(14), to_pixel(16), right_color, 2)

        # 1) 2D side-view
        sim_width, sim_height = 480, 720
        sim_img = 255 * np.ones((sim_height, sim_width, 3), dtype=np.uint8)

        base_point = (sim_width // 2, sim_height // 2 + 50)
        L5 = 100
        L4 = 80
        L3 = 60
        L2 = 0  # disabled in side-view

        m5_rad = math.radians(smoothed_motor5)
        m4_rad = math.radians(smoothed_motor4)
        m3_rad = math.radians(smoothed_motor3)

        p0_local = np.array([0, 0])
        p1_local = p0_local + np.array([L5 * math.cos(m5_rad), L5 * math.sin(m5_rad)])
        p2_local = p1_local + np.array([L4 * math.cos(m5_rad + m4_rad), L4 * math.sin(m5_rad + m4_rad)])
        p3_local = p2_local + np.array([L3 * math.cos(m5_rad + m4_rad + m3_rad),
                                        L3 * math.sin(m5_rad + m4_rad + m3_rad)])
        p4_local = p3_local

        p0 = np.array([base_point[0] + p0_local[0], base_point[1] - p0_local[1]])
        p1 = np.array([base_point[0] + p1_local[0], base_point[1] - p1_local[1]])
        p2 = np.array([base_point[0] + p2_local[0], base_point[1] - p2_local[1]])
        p3 = np.array([base_point[0] + p3_local[0], base_point[1] - p3_local[1]])
        p4 = p3

        p0 = (int(p0[0]), int(p0[1]))
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        p3 = (int(p3[0]), int(p3[1]))
        p4 = (int(p4[0]), int(p4[1]))

        cv2.circle(sim_img, p0, 10, (0, 0, 0), -1)
        cv2.line(sim_img, p0, p1, (255, 0, 0), 4)
        cv2.line(sim_img, p1, p2, (255, 0, 0), 4)
        cv2.line(sim_img, p2, p3, (255, 0, 0), 4)
        cv2.line(sim_img, p3, p4, (255, 0, 0), 4)

        if claw_state_sim == "OPEN":
            cv2.line(sim_img, p4, (p4[0] - 15, p4[1] - 15), (0, 0, 255), 4)
            cv2.line(sim_img, p4, (p4[0] + 15, p4[1] - 15), (0, 0, 255), 4)
        else:
            cv2.rectangle(sim_img, (p4[0] - 10, p4[1] - 10),
                          (p4[0] + 10, p4[1] + 10), (0, 0, 255), -1)

        cv2.putText(sim_img, f"M5 (Shoulder): {int(smoothed_motor5)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sim_img, f"M4 (Elbow): {int(smoothed_motor4)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sim_img, f"M3 (Wrist): {int(smoothed_motor3)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sim_img, f"M1 (Claw): {claw_state_sim}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 2) NEW "3D" PREVIEW: top-down for Motor6=shoulder->elbow, Motor2=elbow->wrist
        sim_width_3d  = 260
        sim_height_3d = 720
        sim_img_3d = 255 * np.ones((sim_height_3d, sim_width_3d, 3), dtype=np.uint8)

        base_point_3d = (sim_width_3d // 2, sim_height_3d - 150)

        # We'll treat Motor6 as the 1st segment angle
        # We'll treat Motor2 as the 2nd segment angle (relative)
        # Segment lengths for top-down:
        segment_len_6 = 100  # "shoulder->elbow"
        segment_len_2 = 60   # "elbow->wrist"

        m6_rad = math.radians(smoothed_motor6)  # 0..360
        # p0_3d = base
        p0_3d = base_point_3d
        # p1_3d => after segment_len_6 at angle m6_rad
        p1_3d = (int(p0_3d[0] + segment_len_6 * math.cos(m6_rad)),
                 int(p0_3d[1] + segment_len_6 * math.sin(m6_rad)))

        # p2_3d => from p1_3d, add segment_len_2 at angle m6_rad + m2_rad (if you want relative),
        # or if we want absolute, just use m2_rad alone. We'll do relative so it doesn't appear "backwards."
        # angle = m6_rad + m2_rad
        angle_rel = m6_rad
        p2_3d = (int(p1_3d[0] + segment_len_2 * math.cos(angle_rel)))

        # Draw in top-down
        cv2.circle(sim_img_3d, p0_3d, 8, (0,0,0), -1)  # base
        cv2.line(sim_img_3d, p0_3d, p1_3d, (255,0,0), 4)  # motor6 segment

        # Label angles
        cv2.putText(sim_img_3d, f"Motor6: {int(smoothed_motor6)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Combine side panel + top-down horizontally
        combined_left = np.hstack((sim_img, sim_img_3d))

        # Then combine with camera feed
        cam_feed = cv2.resize(frame, (640, 720))
        final_combined = np.hstack((combined_left, cam_feed))

        cv2.imshow("Mechanical Arm Simulation (Side + Top-down) and Camera Feed", final_combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()