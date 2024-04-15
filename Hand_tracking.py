import mediapipe as mp
import cv2

# Webcam
cap = cv2.VideoCapture(0)

# Get the Hand Module
mpHand = mp.solutions.hands
Hand   = mpHand.Hands()

# Get the Drawing Module 
mpDraw = mp.solutions.drawing_utils

while cap.isOpened():
    # Read the Frame 
    istrue , frame = cap.read()
    
    if istrue == True:
        # Convert the Image BGR 2 RGB
        rgb_img = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        
        # Processing the Frame
        res = Hand.process(rgb_img)
        
        # Check the Hand in the Frame
        if res.multi_hand_landmarks:
            for hand_in_frame in res.multi_hand_landmarks:
                # Draw Landmarks
                mpDraw.draw_landmarks(frame , hand_in_frame , mpHand.HAND_CONNECTIONS
                                     , mpDraw.DrawingSpec(color = (0,0,0) , thickness = 2 , circle_radius = 3))
                # Find the Coordinate of the Landmarks
                for index , coordinate in enumerate(hand_in_frame.landmark):
                    # Get the Shape of the Frame
                    h , w , c = frame.shape
                    cx , cy   = int(coordinate.x * w) , int(coordinate.y * h)
                    # Draw Circle index 0 
                    if index == 0:
                        cv2.circle(frame , (cx , cy) , 5 , color = (100,200,234) , thickness = 2)
        
        # Flip the frame horizontally to disable the mirror effect
        frame = cv2.flip(frame, 1)
        
        # Display the Frame
        cv2.imshow('Webcam' , frame)
        
        if cv2.waitKey(5) & 0xff == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
