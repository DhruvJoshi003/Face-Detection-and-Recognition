    # Face Recognition using FaceNet and SVM

    This repository contains code for a face recognition system using MTCNN for face detection, FaceNet for generating face embeddings, and an SVM model for classification. The code includes steps for manual preprocessing, automatic preprocessing, face embedding generation, SVM model training, and real-time face recognition using a webcam.

    ## Table of Contents
        - Installation
        - Directory Structure
        - Paths to Update
        - Usage
        - Model Training
        - Real-time Face Recognition
        - Contributing
        - License
    
    ## Installation
    
        1. Clone the repository:
           ```bash
           git clone https://github.com/yourusername/face-recognition-facenet.git
           cd face-recognition-facenet
           ```
        2. Install the required Python packages:
           ```bash
           pip install -r requirements.txt
           ```
    
    ## Directory Structure
    
        ```
        face-recognition-facenet/
        |-- Face_Recognition_using_Facenet.ipynb
        |-- requirements.txt
        |-- README.md
        |-- data/
            |-- Testing_FaceRec/
            |-- Celebrity_Faces_Dataset/
        |-- models/
            |-- face-recognition-model.pkl
        |-- utils/
            |-- face_loading.py
            |-- svm_model.py
        ```
    
    ## Paths to Update
    
        Before running the code, ensure the following paths are correctly updated to your local directories:
        
        1. **Image path for manual preprocessing:**
           ```python
           img = cv.imread('/content/drive/MyDrive/Testing_FaceRec/DJ_test1.jpg')
           ```
           Update to the path of the image you want to test.
        
        2. **Directory for automatic preprocessing:**
           ```python
           faceloading = FaceLoading("/content/drive/MyDrive/Celebrity_Faces_Dataset")
           ```
           Update to the path of your dataset directory.
        
        3. **Model loading path in real-time prediction:**
           ```python
           with open('face-recognition-model.pkl', 'rb') as f:
           ```
           Ensure the model path points to where `face-recognition-model.pkl` is saved.

    ## Usage

        ### Manual Preprocessing
        
        1. Convert the input image from BGR to RGB.
        2. Detect faces using MTCNN.
        3. Crop and resize the detected face for embedding generation.
        
        ### Automatic Preprocessing
        
        - Use the `FaceLoading` class to load and preprocess all images in a given directory.
        
        ### FaceNet Embeddings
        
        - Use the `FaceNet` embedder to generate 512-dimensional embeddings for each face.
        
        ### SVM Model Training
        
        - Train an SVM classifier using the embeddings.
        - Save the trained model for later use.
        
        ### Real-time Face Recognition
        
        - Use the webcam to capture real-time video frames.
        - Detect and recognize faces using the trained model.

    ## Model Training

    1. Run the provided notebook to preprocess images and generate embeddings.
    2. Train the SVM model with the generated embeddings.
    3. Save the trained model using `pickle`.

    ## Real-time Face Recognition

    1. Run the code in the notebook to initiate real-time face recognition using the webcam.
    2. Press 'q' to quit the real-time display.

    ## Contributing

    Contributions are welcome! Please open an issue or submit a pull request for any changes or suggestions.

    ## License

    This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

