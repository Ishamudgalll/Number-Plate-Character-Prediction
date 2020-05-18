import cv2
import face_recognition

input_movie = cv2.VideoCapture("C:/Users/LENOVO/Desktop/in_continuation/video2.mp4")
#input_movie = cv2.VideoCapture(0)
#length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

image = face_recognition.load_image_file("C:/Users/LENOVO/Pictures/Camera Roll/1.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

known_faces = [
face_encoding,
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

count = 1
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break
    

    #if ret == True:
     #   cv2.imshow('window-name',frame)
      #  cv2.imwrite("./output/frame%d.jpg" % count, frame)
       # count = count + 1
        #if cv2.waitKey(10) & 0xFF == ord('q'):
         #   break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)     
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Isha Mudgal"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
        cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Write the resulting image to the output video file
    #print("Writing frame {} / {}".format(frame_number, length))
    #output_movie.write(frame)
    cv2.imwrite("./ output_movie/frame%d.jpg" % count, frame)
    count = count + 1
    
    

# All done!
input_movie.release()
cv2.destroyAllWindows()