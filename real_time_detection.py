import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the SVM model
with open('face-recognition-model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the encoder with the specified classes
encoder = LabelEncoder()
encoder.classes_ = np.array([
    'Angelina Jolie',
    'Brad Pitt',
    'Denzel Washington',
    'Hugh Jackman',
    'Jennifer Lawrence',
    'Johnny Depp',
    'Kate Winslet',
    'Leonardo DiCaprio',
    'Megan Fox',
    'Natalie Portman',
    'Nicole Kidman',
    'Robert Downey Jr',
    'Sandra Bullock',
    'Scarlett Johansson',
    'Tom Cruise',
    'Tom Hanks',
    'Will Smith',
    '_Aradhy Mishra',
    '_Ayan Siddiqui',
    '_Dhruv Joshi'
])

# Initialize MTCNN detector and FaceNet embedder
detector = MTCNN()
embedder = FaceNet()

# Define a confidence threshold for "unknown" classification
confidence_threshold = 0.6 

def process_frame(frame):
    # Detect faces in the frame
    result = detector.detect_faces(frame)
    
    for face in result:
        x, y, w, h = face['box']
        
        # Extract and preprocess the face
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (160, 160))
        
        # Get embeddings for the face
        face_embedding = embedder.embeddings(np.expand_dims(face_img, axis=0))[0].reshape(1, -1)
        
        # Predict the label and get the confidence score
        y_pred = model.predict(face_embedding)
        y_prob = model.predict_proba(face_embedding).max()  # Get the highest probability for the prediction

        print(y_prob)

        # Check if the confidence is below the threshold
        if y_prob < confidence_threshold:
            predicted_label = "unknown"
        else:
            predicted_label = encoder.inverse_transform(y_pred)[0]
        
        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Video capture: 0 for webcam or a video file path
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV format) to RGB for processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Convert RGB back to BGR for OpenCV display
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Display the frame in a window
        cv2.imshow('Real-time Face Recognition', processed_frame)
        
        # Press 'q' to exit the real-time display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()




