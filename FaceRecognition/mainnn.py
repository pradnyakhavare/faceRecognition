import numpy as np
import cv2
import os
import pickle

video = cv2.VideoCapture(0)  # Open webcam

#is used in OpenCV to load a pre-trained Haar cascade classifier for face detection.
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  

face_data = []
name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:
        continue  # Skip if frame is not captured properly

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]  # Extract the face region
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize to (50,50)

        if len(face_data) < 50:  # Stop at 50 images
            face_data.append(resized_img)
            cv2.putText(frame, f"Captured: {len(face_data)}/50", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) == 50:
        break  # Stop when 50 images are captured

video.release()
cv2.destroyAllWindows()

# Convert face_data to numpy array
face_data = np.array(face_data).reshape(50, -1)  # 50 images, flattened

# Ensure the 'data/' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Save names
names_file = 'data/names.pkl'
if not os.path.exists(names_file):
    names = [name] * 50
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * 50)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save face data
face_data_file = 'data/face_data.pkl'
if not os.path.exists(face_data_file):
    with open(face_data_file, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(face_data_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.vstack((faces, face_data))
    with open(face_data_file, 'wb') as f:
        pickle.dump(faces, f)

print("Face data successfully saved!")