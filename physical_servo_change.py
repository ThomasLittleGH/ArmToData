#!/usr/bin/env python3

import xarm
import threading
import time

# Initialize xarm controller
arm = xarm.Controller("USB")

# Store servo data in a dictionary for easy lookup
# If you only need IDs 1,3,4,5,6, just include those.
servos = {
    1: {
        "Servo": xarm.Servo(1),
        "StopEvent": threading.Event(),
        "Thread": None
    },
    3: {
        "Servo": xarm.Servo(3),
        "StopEvent": threading.Event(),
        "Thread": None
    },
    4: {
        "Servo": xarm.Servo(4),
        "StopEvent": threading.Event(),
        "Thread": None
    },
    5: {
        "Servo": xarm.Servo(5),
        "StopEvent": threading.Event(),
        "Thread": None
    },
    6: {
        "Servo": xarm.Servo(6),
        "StopEvent": threading.Event(),
        "Thread": None
    }
}

# Optional reference: (-125.0 to 125.0 in 0.25° increments)
# Not enforced in this code, but you can add checks if desired.

def incremental_move(servo_dict, target_angle, total_ms=1000, steps=20):
    """
    Moves the servo to the target angle in small increments.
    Allows an ongoing move to be stopped by checking StopEvent.
    """
    stop_event = servo_dict["StopEvent"]
    servo = servo_dict["Servo"]
    servo_id = servo.servo_id
    
    # Current angle as reported by xarm (use .angle or .position as needed)
    # If servo.angle is not accurate, consider servo.position or manual tracking.
    current_angle = servo.angle  
    
    angle_step = (target_angle - current_angle) / steps
    step_ms = total_ms / steps  # duration for each small move in ms
    step_seconds = step_ms / 1000.0
    
    print(f"Starting movement of Servo {servo_id} from {current_angle} to {target_angle}")
    
    for _ in range(steps):
        # Check if stop event is set
        if stop_event.is_set():
            print(f"Movement for servo {servo_id} stopped.")
            return
        current_angle += angle_step
        # Issue non-blocking move
        arm.setPosition(servo_id, current_angle, int(step_ms), wait=False)
        time.sleep(step_seconds)
    
    # Final set to ensure exact target is reached (also non-blocking)
    arm.setPosition(servo_id, target_angle, 100, wait=False)
    print(f"Servo {servo_id} reached target angle {target_angle} (requested).")

def main():
    print("=== XArm Servo Control ===")
    print("Enter a servo ID and an angle, separated by a space (e.g. '3 45').")
    print("IDs valid here: 1, 3, 4, 5, 6. Press Ctrl+C to quit.")
    
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_input.strip():
            continue
        
        parts = user_input.split()
        if len(parts) != 2:
            print("Please enter two values: <servo_id> <angle>")
            continue
        
        # Parse servo_id and angle
        try:
            servo_id = int(parts[0])
            angle = float(parts[1])
        except ValueError:
            print("Invalid input. Please enter numeric servo_id and angle.")
            continue
        
        # Check if the servo_id is valid
        if servo_id not in servos:
            print(f"Servo ID {servo_id} is not defined in this script.")
            continue
        
        # Cancel any existing thread for this servo
        servo_data = servos[servo_id]
        if servo_data["Thread"] and servo_data["Thread"].is_alive():
            print(f"Stopping ongoing movement on Servo {servo_id}...")
            servo_data["StopEvent"].set()         # signal the thread to stop
            servo_data["Thread"].join()          # wait for thread to exit
            servo_data["Thread"] = None          # clear reference
            servo_data["StopEvent"].clear()      # reset the event for future use
        
        # Start a new thread for the new movement
        thread = threading.Thread(
            target=incremental_move,
            args=(servo_data, angle)
        )
        servo_data["Thread"] = thread
        thread.start()

if __name__ == "__main__":
    main()
