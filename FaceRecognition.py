import face_recognition
import cv2
import numpy as np
from datetime import datetime
# CONNECTIONS

# DATA
face_image = "C:\PROJECTO_PRODUCAO\F-SIGI-FACE"
face_1 = face_recognition.load_image_file(face_image)
face_image = "C:\PROJECTO_PRODUCAO\F-SIGI-FACE"
face_2 = face_recognition.load_image_file(face_image)
face_image = "C:\PROJECTO_PRODUCAO\F-SIGI-FACE"
face_3 = face_recognition.load_image_file(face_image)

known_face_encodings = [
    face_1,
    face_2,
    face_3,
]
known_face_names = [
    'name_1',
    'name_2',
    'name_3',  
]
known_users_id = []


# INIT VARIABLES

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# CAPTURE CAMERA, 1 IS FOR MORE THAN 1 CAMS

# to access remotly ex: capture = cv2.VideoCapture('rtsp://192.168.1.64/1'), capture = cv2.VideoCapture('rtsp://username:password@192.168.1.64/1')
capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = capture.read()
    
    # resize frame of video to 1/4 for fast face reco
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # convert image to bgr color
    rgb_small_frame = small_frame[:, :, ::-1]
    
    if process_this_frame:
        # Find all the faces and face encodings in current video frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        
        for face_encoding in face_encodings:
            # Check if face matches
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Desconhecido'
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            dt = datetime.now()
            now = dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            
    process_this_frame = not process_this_frame
    
    # DISPLAY RESULTS
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # DRAW A BOX AROUND THE FACE
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # DRAW LABEL NAME
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 4, bottom - 4), font, 1, (0, 0, 255), 2)
        
    # DISPLAY RESULT IMAGE
    cv2.imshow('Image', frame)
    
    # HIT Q TO QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release handle
capture.release()
cv2.destroyAllWindows()