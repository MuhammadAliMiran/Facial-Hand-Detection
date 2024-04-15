import mediapipe as mp
import cv2
import time


# Webcam 
cap = cv2.VideoCapture(0)

# Get The Face Mesh Module
mpFaceMesh  = mp.solutions.face_mesh
FaceMesh    = mpFaceMesh.FaceMesh() # Create the object of FaceMesh

# Get the Hand Module
mpHand = mp.solutions.hands
Hand   = mpHand.Hands()

# Get The Draw Module
mpDraw = mp.solutions.drawing_utils
DrawSpec = mpDraw.DrawingSpec(thickness = 2 , circle_radius = 3)

cTime = 0 
pTime = 0

while cap.isOpened():
    # Read the Frame
    istrue , frame = cap.read()
    
    if istrue == True:
        
        # Flip The Frame
        frame = cv2.flip(frame , 1)
        
        # Convert the BGR Image to RGB Image
        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        # Process the Image
        result = FaceMesh.process(rgb)
        
        # Processing the Frame
        res = Hand.process(rgb)
    
        # Check the Face in the Frame
        if result.multi_face_landmarks:
            for face_in_frame in result.multi_face_landmarks:
                mpDraw.draw_landmarks(frame , face_in_frame , mpFaceMesh.FACEMESH_CONTOURS , DrawSpec)
                # Find the Coordinate of the Face Mesh
                for index , coordinate in enumerate(face_in_frame.landmark):
                    # Get the Shape of the Frame
                    h , w , c = frame.shape
                    cx , cy   = int(coordinate.x * w) , int(coordinate.y * h)

                    if index == 0:
                        cv2.circle(frame , (cx,cy) , 5 , color = (0,0,0) , thickness = -1)
                        
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
        # Display the Frame
        cv2.imshow('Face Mesh' , frame)

        if cv2.waitKey(5) & 0xff == ord('q'):

            break

    else:

        break


cap.release()

cv2.destroyAllWindows()