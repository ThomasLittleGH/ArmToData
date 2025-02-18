#!/usr/bin/env python3

import xarm

def main():
    # Adjust this range to cover all possible IDs you want to probe
    ID_RANGE = range(1, 7)  # For example, 1 through 6
    
    # Initialize the xArm USB controller
    arm = xarm.Controller("USB")
    
    print("=== Probing Servos ===")
    print(f"Checking servo IDs in {list(ID_RANGE)}...\n")

    for i in ID_RANGE:
        try:
            # Attempt to create a servo object
            servo = xarm.Servo(i)
            
            # Query some properties to ensure it responds
            # Depending on the xarm version, you might have .angle, .position, etc.
            current_angle = servo.angle
            current_position = servo.position
            
            print(f"Servo ID {i} is responding!")
            print(f"  Angle: {current_angle}")
            print(f"  Position: {current_position}")
            
            # (Optional) Make a small move to confirm which physical servo it is
            # BE CAREFUL: This will move the servo a few degrees for identification.
            # You can comment this out if you don't want any movement.
            # arm.setPosition(i, current_angle + 5, 200, wait=False)

        except Exception as e:
            # If no servo is found or an error occurs, print it out
            print(f"Servo ID {i} not responding or error encountered: {e}")

    print("\n=== Probe Complete ===")
    print("Check above which IDs responded. Fill those IDs in your main script accordingly.")

if __name__ == "__main__":
    main()
