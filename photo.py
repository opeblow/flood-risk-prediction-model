import numpy as np
def degrees_to_radians(degrees):
    return np.radians(degrees)
def creating_rotation_matrix(roll,pitch,yaw):
    Rx=np.array([1,0,0],
                [0,np.cos(roll),-np.sin(roll)],
                [0,np.sin(roll),np.cos(roll)])
    Ry=np.array([
        [np.cos(pitch),0,np.sin(pitch)],
        [0,1,0],
        [-np.sin(pitch),0,np.cos(pitch)]
    ])
    Rz=np.array([
        [np.cos(yaw),-np.sin(yaw),0],
        [np.sin(yaw),np.cos(yaw),0],
        [0,0,1]
    ])
    return Rx * Ry * Rz
while True:
    try:
        roll_deg=float(input("Please enter Roll(degrees):"))
        pitch_deg=float(input("Please enter Pitch(degrees):"))
        yaw_deg=float(input("Please enter yaw(degrees):"))
        #convert to radians
        roll=degrees_to_radians(roll_deg)
        pitch=degrees_to_radians(pitch_deg)
        yaw=degrees_to_radians(yaw_deg)
        #Get point coordinates from user
        x=float(input("Enter X :"))
        y=float(input("Enter Y :"))
        z=float(input("Enter Z :"))
        point=np.array([x,y,z])
        print(f"DEBUG:point created successfully:{point}")
        print(f"DEBUG:Type of point:{type(point)}")
        print(f"DEBUG:Shape of point:{point.shape}")
        R=creating_rotation_matrix(roll,pitch,yaw)
        rotated_point=R *point
        print("\nRotation Matrix:\n",R)
        print("Original Point:", point)
        print("Rotated Point:",np.round(rotated_point,3))
        questionaire=input("\nRotate another point?(y/n):")
        if questionaire.lower() !='y':
            print("Goodbye!")
            break
    except Exception as e:
        print("Error:",e)
        print("Please enter valid numbers.")

