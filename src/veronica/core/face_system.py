import os
from typing import Optional, Tuple
import traceback
import cv2
import numpy as np
from scipy.spatial import distance
import dlib
import mediapipe as mp
import time 
import pyodbc
import logging
import uuid
import torch
from PIL import Image
from insightface.app import FaceAnalysis
import onnxruntime as ort
from skimage import feature

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting Face Recognition System")

def connect_to_database():
    """Create a connection to the SQL Server database."""
    conn_str = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=172.16.15.161;DATABASE=AI;UID=aiuser;PWD=AIP@ss0rdSQL;TrustServerCertificate=yes;Encrypt=no;'
   
    try:
        connection = pyodbc.connect(conn_str)
        logging.info("Successfully connected to the database!")
        return connection
    except pyodbc.Error as e:
        logging.error(f"Error connecting to the database: {e}")
        raise

def create_face_embeddings_table(connection):
    """Create the face embeddings table if it doesn't exist."""
    try:
        cursor = connection.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='tblFaceEmbeddings' AND xtype='U')
            CREATE TABLE tblFaceEmbeddings (
                FaceID VARCHAR(36) PRIMARY KEY,
                PersonName NVARCHAR(100) NOT NULL,
                FaceEmbeddings VARBINARY(MAX) NOT NULL,
                Created_at DATETIME DEFAULT GETDATE()
            )
        """)
        connection.commit()
        logging.info("Face embeddings table created or verified!")
    except pyodbc.Error as e:
        logging.error(f"Error creating table: {e}")
        raise

class FaceRecognition:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        
        # Initialize InsightFace with recommended settings
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # InsightFace recommended threshold (cosine similarity)
        self.recognition_threshold = 0.50 
        
        # Cache parameters
        self.batch_size = 1000
        self.cache_update_interval = 300  # 5 minutes
        self.last_cache_update = time.time()
        
        # Initialize embeddings storage
        self.embeddings_matrix = None
        self.names_list = []
        self.face_ids = []
        
        # Load existing embeddings
        self.load_embeddings_from_db()

    def extract_face_embeddings(self, face_input):
        """Extract face embeddings using InsightFace."""
        try:
            # Convert tensor or PIL image to numpy array if needed
            if isinstance(face_input, torch.Tensor):
                face_input = face_input.cpu().numpy().transpose(1, 2, 0)
                face_input = (face_input * 255).astype(np.uint8)
            elif isinstance(face_input, Image.Image):
                face_input = np.array(face_input)
            
            # Ensure BGR format for InsightFace
            if len(face_input.shape) == 3 and face_input.shape[2] == 3:
                if face_input.dtype != np.uint8:
                    face_input = (face_input * 255).astype(np.uint8)
                # Only convert to BGR if input is RGB
                if isinstance(face_input, Image.Image) or isinstance(face_input, torch.Tensor):
                    face_input = cv2.cvtColor(face_input, cv2.COLOR_RGB2BGR)
            
            # Log shape for debugging
            logging.info(f"Face input shape: {face_input.shape}")
            
            # Ensure minimum size for face detection
            min_size = 128
            if face_input.shape[0] < min_size or face_input.shape[1] < min_size:
                scale = min_size / min(face_input.shape[0], face_input.shape[1])
                new_size = (int(face_input.shape[1] * scale), int(face_input.shape[0] * scale))
                face_input = cv2.resize(face_input, new_size, interpolation=cv2.INTER_LINEAR)
                logging.info(f"Resized face input to: {face_input.shape}")
            
            # Get face embeddings using InsightFace
            faces = self.app.get(face_input)
            if not faces:
                logging.warning(f"No face detected in input image of shape {face_input.shape}")
                return None
            
            # Get the largest face if multiple faces detected
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Get face embedding directly from the detected face
            embedding = largest_face.embedding
            
            # L2 normalization of embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error in extract_face_embeddings: {str(e)}")
            traceback.print_exc()
            return None

    def load_embeddings_from_db(self):
        """Load face embeddings from database into memory efficiently."""
        try:
            cursor = self.db_connection.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM tblFaceEmbeddings")
            total_count = cursor.fetchone()[0]
            print(f"Total embeddings in database: {total_count}")
            
            # Initialize arrays
            embeddings_list = []
            self.names_list = []
            self.face_ids = []
            
            # Fetch embeddings in batches
            offset = 0
            while offset < total_count:
                cursor.execute("""
                    SELECT FaceID, PersonName, FaceEmbeddings 
                    FROM tblFaceEmbeddings 
                    ORDER BY Created_at 
                    OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """, (offset, self.batch_size))
                
                batch = cursor.fetchall()
                if not batch:
                    break
                
                for face_id, person_name, embeddings_blob in batch:
                    # Convert binary blob to numpy array
                    embeddings = np.frombuffer(embeddings_blob, dtype=np.float32)
                    
                    # Ensure correct shape (512,) for InsightFace embeddings
                    if embeddings.size == 512:
                        # Normalize the embedding
                        norm = np.linalg.norm(embeddings)
                        if norm > 0:
                            embeddings = embeddings / norm
                        embeddings_list.append(embeddings)
                        self.names_list.append(person_name)
                        self.face_ids.append(face_id)
                    else:
                        logging.warning(f"Skipping invalid embedding for {person_name} (wrong size: {embeddings.size})")
                
                offset += self.batch_size
                print(f"Loaded {offset} embeddings...")
            
            # Convert to numpy array for efficient computation
            if embeddings_list:
                self.embeddings_matrix = np.vstack(embeddings_list)
                print(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")
            else:
                self.embeddings_matrix = None
            
            self.last_cache_update = time.time()
            logging.info(f"Loaded {len(self.names_list)} face embeddings from database")
            
        except Exception as e:
            logging.error(f"Error loading embeddings: {str(e)}")
            traceback.print_exc()
            
            # Initialize empty state on error
            self.embeddings_matrix = None
            self.names_list = []
            self.face_ids = []

    def update_cache_if_needed(self):
        """Check and update cache if interval has passed."""
        current_time = time.time()
        if current_time - self.last_cache_update > self.cache_update_interval:
            self.load_embeddings_from_db()
    
    def add_face_from_id_card(self, face_input, person_name: str):
        """Add a new face to the database and update cache."""
        try:
            # Generate unique FaceID
            face_id = str(uuid.uuid4())
            
            # Extract embeddings
            embeddings = self.extract_face_embeddings(face_input)
            if embeddings is None:
                return False
            
            # Convert embeddings to binary for storage
            embeddings_blob = embeddings.tobytes()
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO tblFaceEmbeddings (FaceID, PersonName, FaceEmbeddings)
                VALUES (?, ?, ?)
            """, (face_id, person_name, pyodbc.Binary(embeddings_blob)))
            self.db_connection.commit()
            
            # Update local cache
            if self.embeddings_matrix is None:
                self.embeddings_matrix = embeddings.reshape(1, -1)
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
            self.names_list.append(person_name)
            self.face_ids.append(face_id)
            
            logging.info(f"Added new face for {person_name} with ID {face_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding face: {str(e)}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face using InsightFace embeddings."""
        try:
            # Check if cache needs updating
            self.update_cache_if_needed()
            
            # Log database state
            logging.info(f"Current database state: {len(self.names_list)} faces stored")

            # Extract embeddings for the input face
            query_embeddings = self.extract_face_embeddings(face_image)
            if query_embeddings is None:
                logging.warning("Could not extract embeddings from input face")
                return "Unknown", 0.0
            
            # If no embeddings in database, return Unknown
            if self.embeddings_matrix is None or len(self.names_list) == 0:
                logging.warning("No faces in database for recognition")
                return "Unknown", 0.0
            
            # Log shapes for debugging
            logging.info(f"Query embedding shape: {query_embeddings.shape}")
            logging.info(f"Database embeddings shape: {self.embeddings_matrix.shape}")
            
            # Calculate cosine similarities with all embeddings
            similarities = np.dot(self.embeddings_matrix, query_embeddings)
            
            # Find best match
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            # Return name and confidence score with more detailed logging
            if max_similarity > self.recognition_threshold:
                logging.info(f"Face recognized as {self.names_list[max_similarity_idx]} with confidence {max_similarity:.4f}")
                return self.names_list[max_similarity_idx], float(max_similarity)
            else:
                logging.info(f"Face not recognized. Best match ({self.names_list[max_similarity_idx]}) score {max_similarity:.4f} below threshold {self.recognition_threshold}")
                return "Unknown", float(max_similarity)
                
        except Exception as e:
            logging.error(f"Error in recognize_face: {str(e)}")
            traceback.print_exc()
            return "Unknown", 0.0

    def register_face_with_multiple_angles(self, face_input: np.ndarray, person_name: str):
        """Register a face using multiple angles and variations for better recognition."""
        try:
            embeddings_list = []
            
            # Get base embedding from original image
            faces = self.app.get(face_input)
            if not faces:
                logging.error("No face detected in original image")
                return False
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            base_embedding = largest_face.embedding
            if base_embedding is not None:
                embeddings_list.append(base_embedding)
            
            # Create variations of the face image
            h, w = face_input.shape[:2]
            center = (w/2, h/2)
            
            # Rotation variations (smaller angles for better accuracy)
            for angle in [-10, -5, 5, 10]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(face_input, M, (w, h))
                faces = self.app.get(rotated)
                if faces:
                    largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                    embedding = largest_face.embedding
                    if embedding is not None:
                        embeddings_list.append(embedding)
            
            # Scale variations
            for scale in [0.95, 1.05]:
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)
                scaled = cv2.resize(face_input, (scaled_w, scaled_h))
                if scale > 1:
                    start_x = (scaled_w - w) // 2
                    start_y = (scaled_h - h) // 2
                    scaled = scaled[start_y:start_y+h, start_x:start_x+w]
                else:
                    pad_x = (w - scaled_w) // 2
                    pad_y = (h - scaled_h) // 2
                    scaled = cv2.copyMakeBorder(scaled, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)
                
                faces = self.app.get(scaled)
                if faces:
                    largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                    embedding = largest_face.embedding
                    if embedding is not None:
                        embeddings_list.append(embedding)
            
            if not embeddings_list:
                logging.error("No valid embeddings generated")
                return False
            
            # Average the embeddings
            avg_embedding = np.mean(embeddings_list, axis=0)
            # Normalize the average embedding
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Store in database
            face_id = str(uuid.uuid4())
            embeddings_blob = avg_embedding.tobytes()
            
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO tblFaceEmbeddings (FaceID, PersonName, FaceEmbeddings)
                VALUES (?, ?, ?)
            """, (face_id, person_name, pyodbc.Binary(embeddings_blob)))
            self.db_connection.commit()
            
            # Update local cache
            if self.embeddings_matrix is None:
                self.embeddings_matrix = avg_embedding.reshape(1, -1)
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, avg_embedding])
            self.names_list.append(person_name)
            self.face_ids.append(face_id)
            
            logging.info(f"Added new face for {person_name} with ID {face_id} using {len(embeddings_list)} variations")
            return True
            
        except Exception as e:
            logging.error(f"Error in multi-angle registration: {str(e)}")
            traceback.print_exc()
            return False

class FaceAligner:
    def __init__(self, desired_face_width=256):
        """Initialize face aligner with dlib's face landmarks predictor"""
        self.desired_face_width = desired_face_width
        self.face_width = desired_face_width
        self.face_height = desired_face_width
        
        # Initialize dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face landmarks and align face."""
        # Validate input image
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            print("Invalid input image: Image is empty or not a numpy array")
            return None
            
        # Check image dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"Invalid image format: Expected 3-channel color image, got shape {image.shape}")
            return None

        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            rects = self.detector(gray, 2)
            if len(rects) == 0:
                print("No faces detected in image")
                return None
                
            # Get facial landmarks
            shape = self.predictor(gray, rects[0])
            coords = np.zeros((68, 2), dtype=np.float32)
            
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            
            # Get the center of each eye
            left_eye = coords[36:42].mean(axis=0)
            right_eye = coords[42:48].mean(axis=0)
            
            # Calculate angle for alignment
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Calculate desired right eye position
            desired_right_eye_x = 1.0 - 0.35
            
            # Calculate the scale of the new resulting image
            dist = distance.euclidean(left_eye, right_eye)
            desired_dist = (desired_right_eye_x - 0.35) * self.face_width
            scale = desired_dist / dist
            
            # Calculate center of eyes
            eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                         (left_eye[1] + right_eye[1]) // 2)
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            
            # Update the translation component of the matrix
            tX = self.face_width * 0.5
            tY = self.face_height * 0.35
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])
            
            # Perform affine transformation
            aligned_face = cv2.warpAffine(image, M, (self.face_width, self.face_height),
                                        flags=cv2.INTER_CUBIC)
            
            return aligned_face
            
        except Exception as e:
            print(f"Error in face alignment: {str(e)}")
            traceback.print_exc()
            return None


class IrisDepthEstimator:
    def __init__(self):
        """Initialize iris depth estimator using MediaPipe Face Mesh."""
        # Initialize MediaPipe Face Landmarker
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Download model if not exists
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            os.system('wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
        
        # Create face landmarker for video mode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.timestamp = 0
        self.last_detection_result = None
        
        # Reference measurements (in millimeters)
        self.IRIS_DIAMETER_MM = 11.7  # Average human iris diameter
        self.FOCAL_LENGTH_NORM = 1.40625  # Normalized focal length
        
        # MediaPipe Face Mesh landmark indices for iris
        self.LEFT_IRIS = [468, 469, 470, 471, 472]   # Center, right, top, left, bottom
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]  # Center, right, top, left, bottom
        
        # Temporal smoothing
        self.depth_history = []
        self.max_history = 5  # Reduced from 10 to 5 for faster response
        self.last_valid_depth = None
        
        # Motion analysis parameters
        self.last_landmarks = None
        self.last_depth = None
        self.motion_history = []
        self.max_motion_history = 5  # Reduced from 10 to 5
        self.micro_movement_threshold = 0.002  # Increased from 0.001
        self.max_movement = 0.15  # Increased from 0.1
        self.landmark_groups = {
            'left_eye': list(range(33, 246)),
            'right_eye': list(range(362, 466)),
            'mouth': list(range(0, 17)),
            'nose': list(range(168, 197)),
        }
        self.differential_weights = {
            'left_eye': 1.2,    # Reduced from 1.5
            'right_eye': 1.2,   # Reduced from 1.5
            'mouth': 0.8,       # Reduced from 1.0
            'nose': 0.6,        # Reduced from 0.8
        }

    def get_iris_landmarks(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract iris landmarks using MediaPipe Face Landmarker."""
        if frame is None or frame.size == 0:
            return None
            
        try:
            # Convert to MediaPipe Image format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame)
            )
            
            # Process the frame
            self.last_detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp)
            self.timestamp += 1
            
            if not self.last_detection_result.face_landmarks:
                return None
                
            face_landmarks = self.last_detection_result.face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Extract iris landmarks relative to the face region
            left_iris = np.array([[face_landmarks[idx].x * w,
                                 face_landmarks[idx].y * h]
                                for idx in self.LEFT_IRIS])
            
            right_iris = np.array([[face_landmarks[idx].x * w,
                                  face_landmarks[idx].y * h]
                                for idx in self.RIGHT_IRIS])
            
            return left_iris, right_iris
            
        except Exception as e:
            print(f"Error in iris landmark detection: {str(e)}")
            return None
    
    def estimate_depth(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Estimate depth using iris diameter."""
        iris_landmarks = self.get_iris_landmarks(frame)
        if not iris_landmarks:
            return None
            
        try:
            left_iris, right_iris = iris_landmarks
            
            # Calculate iris diameters (using horizontal width)
            left_diameter = np.max(left_iris[:, 0]) - np.min(left_iris[:, 0])
            right_diameter = np.max(right_iris[:, 0]) - np.min(right_iris[:, 0])
                
            # Calculate confidence based on diameter consistency
            diameter_ratio = min(left_diameter, right_diameter) / max(left_diameter, right_diameter)
            measurement_confidence = diameter_ratio * diameter_ratio  # Square for more aggressive penalty
                
            # Use average diameter
            iris_diameter_pixels = (left_diameter + right_diameter) / 2
                
            # Calculate depth using focal length
            focal_length = min(frame.shape[0], frame.shape[1]) * self.FOCAL_LENGTH_NORM
            depth_mm = (focal_length * self.IRIS_DIAMETER_MM) / iris_diameter_pixels
            
            # Apply temporal smoothing with less history
            self.depth_history.append(depth_mm)
            if len(self.depth_history) > self.max_history:
                self.depth_history.pop(0)
            
            # Use median for final depth with more weight on recent measurements
            weights = np.linspace(0.5, 1.0, len(self.depth_history))
            smoothed_depth = np.average(self.depth_history, weights=weights)
            
            # Calculate final confidence with less penalty for variation
            temporal_confidence = 1.0 / (1.0 + np.std(self.depth_history) / 20.0)  # Increased from 10.0 to 20.0
            final_confidence = (measurement_confidence + temporal_confidence) / 2  # Average instead of multiply
                
            self.last_valid_depth = smoothed_depth
            return smoothed_depth, final_confidence
            
        except Exception as e:
            print(f"Error in depth estimation: {str(e)}")
            return None
    
    def draw_debug_info(self, frame: np.ndarray, depth_info: Optional[Tuple[float, float]] = None):
        """Draw debug visualization."""
        if frame is None or frame.size == 0:
            return frame

        try:
            # Get frame dimensions first
            h, w = frame.shape[:2]
            
            # Draw depth information in top-right corner
            if depth_info:
                depth_mm, confidence = depth_info
                font_scale = h / 720.0  # Adaptive font scale
                thickness = max(1, int(h / 500))  # Adaptive thickness
                line_height = int(30 * font_scale)
                
                # Position text in top-right corner with padding
                padding = 10
                x_pos = w - 200  # Fixed distance from right edge
                y_pos = padding + line_height
                
                # Draw depth measurement
                cv2.putText(frame, f"Depth: {depth_mm:.1f}mm", 
                           (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (0, 255, 255), thickness)
                
                # Draw confidence score
                y_pos += line_height
                cv2.putText(frame, f"Conf: {confidence:.2f}",
                           (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (0, 255, 255), thickness)
            
            # Use the last detection result instead of processing again
            if self.last_detection_result and self.last_detection_result.face_landmarks:
                # Draw face landmarks
                face_landmarks = self.last_detection_result.face_landmarks[0]
                
                # Define key facial feature indices
                FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
                
                # Draw only key facial features with small white dots
                for idx in FACE_OUTLINE + LEFT_EYE + RIGHT_EYE + LIPS:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)
                
                # Draw iris landmarks with larger, more visible points
                for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    # Draw larger circles for iris landmarks
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dots
                    cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)      # Black outline
                
                # Draw connections between iris landmarks
                def draw_iris_connections(iris_points):
                    for i in range(1, 5):
                        pt1 = (int(face_landmarks[iris_points[i]].x * w),
                              int(face_landmarks[iris_points[i]].y * h))
                        pt2 = (int(face_landmarks[iris_points[0]].x * w),
                              int(face_landmarks[iris_points[0]].y * h))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
                
                draw_iris_connections(self.LEFT_IRIS)
                draw_iris_connections(self.RIGHT_IRIS)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing debug info: {str(e)}")
            return frame


# Anti-spoofing settings
COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)
SPOOF_THRESHOLD = float(os.getenv('SPOOF_THRESHOLD', '0.7'))

class DeepAntiSpoof:
    def __init__(self, weights: str = 'models/AntiSpoofing_bin_1.5_128.onnx', model_img_size: int = 128):
        """Initialize the deep anti-spoofing model."""
        self.weights = weights
        self.model_img_size = model_img_size
        
        # Ensure model file exists
        if not os.path.exists(weights):
            # Try to find the model in the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, weights)
            if not os.path.exists(weights_path):
                logging.error(f"Model file not found at {weights} or {weights_path}")
                raise FileNotFoundError(f"Model file not found at {weights} or {weights_path}")
            self.weights = weights_path
        
        # Initialize ONNX session with proper error handling
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']  # Start with CPU provider
            
            # Try to initialize with CUDA if available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider'] + providers
                logging.info("CUDA is available for anti-spoofing model")
            
            # Create session options
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 1
            
            # Initialize session
            self.ort_session = ort.InferenceSession(
                self.weights,
                providers=providers,
                sess_options=options
            )
            
            # Get input and output details
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            
            # Log successful initialization
            logging.info(f"Anti-spoofing model initialized successfully with:")
            logging.info(f"- Model path: {self.weights}")
            logging.info(f"- Providers: {providers}")
            logging.info(f"- Input name: {self.input_name}")
            logging.info(f"- Output name: {self.output_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize anti-spoofing model: {str(e)}")
            raise RuntimeError(f"Could not initialize anti-spoofing model: {str(e)}")

    def __call__(self, imgs: list):
        """Run inference on a list of images."""
        if not isinstance(imgs, list):
            raise ValueError("Input must be a list of images")
        
        if self.ort_session is None:
            raise RuntimeError("Model not properly initialized")
        
        try:
            preds = []
            for img in imgs:
                if img is None or img.size == 0:
                    raise ValueError("Invalid image in input list")
                
                preprocessed = self.preprocessing(img)
                onnx_result = self.ort_session.run(
                    [self.output_name],  # Specify output name
                    {self.input_name: preprocessed}
                )
                pred = self.postprocessing(onnx_result[0])
                preds.append(pred)
            return preds
            
        except Exception as e:
            logging.error(f"Error in model inference: {str(e)}")
            raise

    def preprocessing(self, img):
        """Preprocess an image for the model."""
        if img is None or img.size == 0:
            raise ValueError("Invalid input image")
            
        try:
            # Resize image
            img = cv2.resize(img, (self.model_img_size, self.model_img_size))
            
            # Convert to float32 and normalize
            img = img.astype(np.float32) / 255.0
            
            # Transpose to channel-first format (NCHW)
            img = img.transpose(2, 0, 1)
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def postprocessing(self, prediction):
        """Apply softmax to model predictions."""
        try:
            # Compute softmax
            exp_preds = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
            softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            return softmax_preds
            
        except Exception as e:
            logging.error(f"Error in postprocessing: {str(e)}")
            raise

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                            y1-y, int(l*bbox_inc-y2+y),
                            x1-x, int(l*bbox_inc)-x2+x,
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

class AntiSpoofing:
    def __init__(self):
        """Initialize anti-spoofing system with dynamic constraints."""
        # Layer 1: Deep learning model
        self.deep_model = DeepAntiSpoof(weights='models/AntiSpoofing_bin_1.5_128.onnx', model_img_size=128)
        
        # Base confidence thresholds
        self.confidence_thresholds = {
            'deep_model': 0.50,
            'geometric': 0.50
        }
        
        # Dynamic ratio calculation parameters
        self.position_tolerance = 0.50  # 50% of oval size
        self.size_tolerance = 0.40      # 40% deviation from optimal
        
        # Initialize with None, will be set dynamically
        self.min_face_ratio = None
        self.max_face_ratio = None
        self.optimal_face_ratio = None
        self.depth_size_ratio_tolerance = None
        self.expected_size_at_distance = None

    def calculate_dynamic_constraints(self, oval_guide: dict, frame_height: int):
        """Calculate dynamic constraints based on oval guide and frame dimensions."""
        # Calculate optimal face ratio based on oval dimensions and frame
        oval_height = oval_guide.get('height', 0)
        frame_height = oval_guide.get('frame_height', frame_height)  # Use provided frame height or fallback
        
        if not frame_height or not oval_height:
            logging.error(f"Invalid dimensions - oval height: {oval_height}, frame height: {frame_height}")
            return False
        
        # Log the dimensions we're working with
        logging.info(f"Calculating dynamic constraints - Oval height: {oval_height}, Frame height: {frame_height}")
        
        # Calculate base ratios relative to frame height
        base_ratio = oval_height / frame_height
        logging.info(f"Base ratio (oval/frame): {base_ratio:.3f}")
        
        # Set optimal ratio as the base ratio
        self.optimal_face_ratio = base_ratio
        
        # Calculate min and max ratios with tolerance
        self.min_face_ratio = base_ratio * (1.0 - self.size_tolerance)
        self.max_face_ratio = base_ratio * (1.0 + self.size_tolerance)
        
        logging.info(f"Dynamic ratios - Min: {self.min_face_ratio:.3f}, Optimal: {self.optimal_face_ratio:.3f}, Max: {self.max_face_ratio:.3f}")
        
        # Calculate depth-size correlation based on frame height
        self.expected_size_at_distance = {
            300: base_ratio * 1.3,  # Very close
            400: base_ratio * 1.1,  # Close
            500: base_ratio * 1.0,  # Optimal
            600: base_ratio * 0.9,  # Far
            700: base_ratio * 0.8,  # Very far
            800: base_ratio * 0.7   # Too far
        }
        
        logging.info("Dynamic depth-size correlation:")
        for depth, size in self.expected_size_at_distance.items():
            logging.info(f"  At {depth}mm: Expected size ratio = {size:.3f}")
        
        # Set depth tolerance
        self.depth_size_ratio_tolerance = 0.5  # 50% tolerance for depth-size correlation
        
        return True

    def check_geometric_constraints(self, face_region: tuple, depth_mm: float, 
                                 landmarks: list, oval_guide: dict) -> Tuple[bool, str, float]:
        """Check geometric constraints with dynamic sizing."""
        try:
            if oval_guide is None:
                return False, "Oval guide not provided", 0.0

            x1, y1, x2, y2 = face_region
            face_height = y2 - y1
            face_width = x2 - x1
            
            # Calculate dynamic constraints if not already set
            if self.optimal_face_ratio is None:
                frame_height = oval_guide.get('frame_height')
                if frame_height is None:
                    logging.error("Frame height not provided in oval guide")
                    return False, "Configuration error", 0.0
                    
                success = self.calculate_dynamic_constraints(oval_guide, frame_height)
                if not success:
                    return False, "Failed to calculate constraints", 0.0
            
            # Log the measurements
            logging.info(f"Checking geometric constraints - Face height: {face_height}, Face width: {face_width}")
            
            # Calculate face ratios using both width and height relative to oval
            width_ratio = face_width / oval_guide['width']
            height_ratio = face_height / oval_guide['height']
            face_ratio = min(width_ratio, height_ratio)  # Use smaller ratio to be more lenient
            
            logging.info(f"Face ratios - Width: {width_ratio:.2f}, Height: {height_ratio:.2f}, Used: {face_ratio:.2f}")
            
            if face_ratio < self.min_face_ratio:
                logging.warning(f"Face too small: {face_ratio:.2f} < {self.min_face_ratio:.2f}")
                return False, "Please move closer", 0.0
            if face_ratio > self.max_face_ratio:
                logging.warning(f"Face too large: {face_ratio:.2f} > {self.max_face_ratio:.2f}")
                return False, "Please move back", 0.0
            
            # Check position relative to oval
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            oval_center_x = oval_guide['x'] + oval_guide['width'] / 2
            oval_center_y = oval_guide['y'] + oval_guide['height'] / 2
            
            h_offset = abs(face_center_x - oval_center_x) / oval_guide['width']
            v_offset = abs(face_center_y - oval_center_y) / oval_guide['height']
            
            logging.info(f"Position offsets - Horizontal: {h_offset:.2f}, Vertical: {v_offset:.2f}")
            
            if h_offset > self.position_tolerance or v_offset > self.position_tolerance:
                logging.warning(f"Face position outside acceptable range - H: {h_offset:.2f}, V: {v_offset:.2f}")
                return False, "Please center your face in the oval", 0.0
            
            # Skip depth check if confidence is too low
            if depth_mm is not None and depth_mm > 0:
                depths = list(self.expected_size_at_distance.keys())
                sizes = list(self.expected_size_at_distance.values())
                expected_size = np.interp(depth_mm, depths, sizes)
                size_deviation = abs(face_ratio - expected_size) / expected_size
                
                logging.info(f"Depth correlation - Expected size: {expected_size:.2f}, Actual: {face_ratio:.2f}, Deviation: {size_deviation:.2f}")
                
                if size_deviation > self.depth_size_ratio_tolerance:
                    logging.warning(f"Size deviation too large: {size_deviation:.2f} > {self.depth_size_ratio_tolerance}")
                    return False, "Please adjust your distance to the oval shape", 0.0
            
            # Calculate confidence scores with more weight on position
            position_conf = 1.0 - (h_offset + v_offset) / 2
            size_conf = 1.0 - abs(face_ratio - self.optimal_face_ratio) / (self.max_face_ratio - self.min_face_ratio)
            geometric_conf = (position_conf * 0.8 + size_conf * 0.2)  # More weight on position
            
            logging.info(f"Confidence scores - Position: {position_conf:.2f}, Size: {size_conf:.2f}, Final: {geometric_conf:.2f}")
            
            return True, "Geometric checks passed", geometric_conf

        except Exception as e:
            logging.error(f"Error in geometric checks: {str(e)}")
            return False, "Error in geometric verification", 0.0

    def check_liveness(self, frame: np.ndarray, face_region: tuple, depth_mm: float, 
                      confidence: float, landmarks: list = None, oval_guide: dict = None) -> Tuple[bool, str, float]:
        """Enhanced liveness check with separated deep and geometric verifications."""
        try:
            if oval_guide is None:
                return False, "Oval guide not provided", 0.0

            # 1. Run deep model verification first (on full frame and context)
            deep_ok, deep_msg, deep_conf = self.check_deep_model(frame, face_region)
            
            # 2. Run geometric constraints check
            geo_ok, geo_msg, geo_conf = self.check_geometric_constraints(
                face_region, depth_mm, landmarks, oval_guide
            )

            # Both checks must pass independently
            if not deep_ok:
                return False, deep_msg, deep_conf
            
            if not geo_ok:
                return False, geo_msg, geo_conf

            # If both passed, return success with combined confidence
            final_confidence = min(deep_conf, geo_conf)  # Use minimum as final confidence
            return True, "Face verified", final_confidence

        except Exception as e:
            logging.error(f"Error in liveness check: {str(e)}")
            return False, "Error checking liveness", 0.0

    def draw_feedback(self, frame: np.ndarray, is_live: bool, message: str, 
                     confidence: float, face_region: tuple, depth_mm: float) -> np.ndarray:
        """Draw anti-spoofing feedback on the frame."""
        try:
            height, width = frame.shape[:2]
            font_scale = height / 720.0
            thickness = max(1, int(height / 500))
            
            # Define colors
            success_color = (0, 255, 0)    # Green for positive feedback
            warning_color = (0, 165, 255)  # Orange for warnings
            error_color = (0, 0, 255)      # Red for errors
            
            # Choose color based on confidence and message
            if "Deep model" in message:
                color = success_color if is_live else error_color
            else:
                # Using fallback checks
                color = warning_color if is_live else error_color
            
            # Draw face rectangle
            x1, y1, x2, y2 = face_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw message at the top
            cv2.putText(frame, message,
                       (10, int(30 * font_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       color, thickness)
            
            # Draw confidence
            y_pos = height - int(30 * font_scale)
            cv2.putText(frame, f"Confidence: {confidence:.2f}",
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       color, thickness)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing feedback: {str(e)}")
            return frame

    def check_deep_model(self, frame: np.ndarray, face_region: tuple) -> Tuple[bool, str, float]:
        """Run deep model verification on the full frame with focus on face region."""
        try:
            logging.info("Starting deep model verification check")
            x1, y1, x2, y2 = face_region
            
            # Get the full frame dimensions
            h, w = frame.shape[:2]
            logging.info(f"Frame dimensions: {w}x{h}")
            logging.info(f"Face region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Validate frame and face region
            if frame is None or frame.size == 0:
                logging.error("Empty frame provided to deep model")
                return False, "Position your face in the oval", 0.0
                
            if None in (x1, y1, x2, y2):
                logging.error("Invalid face region coordinates")
                return False, "Position your face in the oval", 0.0

            # Extract face area
            face_area = frame[y1:y2, x1:x2]
            if face_area is None or face_area.size == 0:
                logging.error("Failed to extract face area from frame")
                return False, "Position your face in the oval", 0.0

            # Calculate basic metrics
            face_height = y2 - y1
            face_width = x2 - x1
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            logging.info(f"Face dimensions: width={face_width}, height={face_height}, aspect_ratio={aspect_ratio:.2f}")
            
            # Check if face has reasonable proportions
            if not (0.5 <= aspect_ratio <= 1.0):
                logging.warning(f"Face aspect ratio {aspect_ratio:.2f} outside acceptable range [0.5, 1.0]")
                return False, "Position your face properly", 0.0

            # Calculate face area relative to frame
            face_area_ratio = (face_width * face_height) / (w * h)
            logging.info(f"Face area ratio: {face_area_ratio:.3f}")
            
            if face_area_ratio < 0.02:
                logging.warning("Face too small in frame")
                return False, "Please move closer to the oval shape", 0.0
            if face_area_ratio > 0.5:
                logging.warning("Face too large in frame")
                return False, "Please move back from the oval shape", 0.0

            # Run deep model prediction
            try:
                # Preprocess face area
                face_preprocessed = self.deep_model.preprocessing(face_area)
                
                # Run model prediction
                predictions = self.deep_model.ort_session.run(
                    [self.deep_model.output_name],
                    {self.deep_model.input_name: face_preprocessed}
                )
                
                if not predictions:
                    logging.error("Deep model returned no predictions")
                    return False, "Error in face verification", 0.0
                
                # Apply postprocessing
                predictions = self.deep_model.postprocessing(predictions[0])
                
                # Get confidence score
                spoof_score = predictions[0][0]  # First class probability
                is_real = np.argmax(predictions) == 0  # Class 0 is real face
                confidence = float(spoof_score if is_real else 1.0 - spoof_score)
                
                logging.info(f"Deep model prediction - Is real: {is_real}, Confidence: {confidence:.2f}")
                
                if is_real and confidence > self.confidence_thresholds['deep_model']:
                    return True, "Face verified", confidence
                else:
                    return False, "Please look directly at the camera", confidence
                    
            except Exception as e:
                logging.error(f"Error in deep model prediction: {str(e)}")
                return False, "Error in face verification", 0.0

        except Exception as e:
            logging.error(f"Error in deep model check: {str(e)}")
            traceback.print_exc()
            return False, "Position your face in the oval", 0.0


class FaceSystem:
    def __init__(self, min_detection_confidence=0.5, target_fps=30, input_size=(640, 480)):
        """Initialize the complete face processing system."""
        # Initialize database connection
        self.db_connection = connect_to_database()
        create_face_embeddings_table(self.db_connection)
        
        # Initialize face recognition
        self.face_recognition = FaceRecognition(self.db_connection)
        
        # Download required models if they don't exist
        if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
            print("Downloading face landmarks model...")
            os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
            os.system('bzip2 -d shape_predictor_68_face_landmarks.dat.bz2')

        # Initialize components with optimized parameters
        self.face_aligner = FaceAligner(desired_face_width=160)
        self.depth_estimator = IrisDepthEstimator()
        self.anti_spoofing = AntiSpoofing()
        
        # Performance optimization parameters
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0
        self.input_size = input_size
        self.skip_frames = 0
        
        # Adjust depth and face size constraints
        STANDARD_FACE_WIDTH_MM = 150.0  # Average human face width
        OPTIMAL_DEPTH_MM = 400.0  # Increased from 300mm to 400mm
        DEPTH_RANGE = 0.4  # Increased from 0.3 to 0.4 (40% deviation allowed)

        # Initialize InsightFace with more lenient settings
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.recognition_threshold = 0.45  # Decreased from 0.50

        # Processing flags
        self.process_alignment = True
        self.process_depth = True
        
        # Performance metrics
        self.fps_history = []
        self.processing_times = []
        
        # System state
        self.last_depth_mm = None
        self.last_confidence = None
        self.last_liveness = None
        self.last_liveness_message = None
        self.processing_stats = {
            'faces_detected': 0,
            'successful_alignments': 0,
            'depth_measurements': 0,
            'spoof_attempts': 0,
            'avg_fps': 0,
            'avg_processing_time': 0
        }

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better face detection."""
        # Resize frame to target size while maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize using INTER_LINEAR for better speed
        processed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Skip CLAHE for better performance
        return processed

    def process_frame(self, frame: np.ndarray, draw_debug: bool = True, oval_guide: dict = None) -> Tuple[np.ndarray, dict]:
        """Process a single frame through the entire face processing pipeline."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logging.error("Invalid frame provided to process_frame")
            return frame, {}

        try:
            logging.info("Starting frame processing")
            start_time = time.time()
            
            # Skip frames if we're falling behind
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                self.skip_frames += 1
                logging.debug(f"Skipping frame {self.skip_frames} due to FPS limit")
                return frame, {}
            self.last_frame_time = current_time

            # Preprocess frame
            logging.info("Preprocessing frame")
            processed_frame = self.preprocess_frame(frame)
            logging.info(f"Preprocessed frame dimensions: {processed_frame.shape}")

            # Initialize results dictionary
            results = {
                'success': True,
                'is_live': False,
                'liveness_confidence': 0.0,
                'liveness_message': "Position your face within the oval",
                'recognized_name': None,
                'new_exposure': None,
                'face_detected': False,
                'face_position': None,
                'depth_valid': False
            }

            # Create visualization frame only if needed
            display_frame = frame.copy() if draw_debug else frame

            # Get face landmarks first
            logging.info("Detecting face landmarks")
            face_landmarks = self.depth_estimator.get_iris_landmarks(processed_frame)
            if not face_landmarks or not self.depth_estimator.last_detection_result:
                logging.warning("No face landmarks detected")
                results['liveness_message'] = "No face detected - Please look at the camera"
                return display_frame, results

            # Check for multiple faces
            num_faces = len(self.depth_estimator.last_detection_result.face_landmarks)
            logging.info(f"Detected {num_faces} faces in frame")
            if num_faces > 1:
                logging.warning("Multiple faces detected")
                results['liveness_message'] = "Multiple faces detected - Please ensure only one person is in frame"
                return display_frame, results

            # Get face position from landmarks
            landmarks = self.depth_estimator.last_detection_result.face_landmarks[0]
            h, w = frame.shape[:2]
            logging.info(f"Original frame dimensions: {w}x{h}")
            
            # Get all landmark points
            points = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] 
                             for idx in range(468)])
            logging.info(f"Extracted {len(points)} facial landmarks")

            # Calculate face region with padding
            x1, y1 = np.min(points, axis=0).astype(int)
            x2, y2 = np.max(points, axis=0).astype(int)
            logging.info(f"Initial face bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Add padding
            padding_w = int((x2 - x1) * 0.3)
            padding_h = int((y2 - y1) * 0.3)
            logging.info(f"Adding padding: width={padding_w}, height={padding_h}")
            
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(w, x2 + padding_w)
            y2 = min(h, y2 + padding_h)
            logging.info(f"Final face bounds with padding: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            face_width = x2 - x1
            face_height = y2 - y1

            # Check face aspect ratio
            face_aspect_ratio = face_width / face_height if face_height > 0 else 0
            if not (0.8 <= face_aspect_ratio <= 1.2):
                logging.warning(f"Invalid face aspect ratio: {face_aspect_ratio:.2f}")
                results['liveness_message'] = "Please face the camera directly"
                return display_frame, results

            # Get depth estimation
            depth_mm = None
            depth_confidence = None
            if self.process_depth:
                depth_info = self.depth_estimator.estimate_depth(processed_frame)
                if depth_info is not None:
                    depth_mm, depth_confidence = depth_info
                    
                    # Validate depth measurement
                    if abs(depth_mm - 500) > 100:
                        if depth_mm < 500 - 100:
                            results['liveness_message'] = "Please move back"
                        else:
                            results['liveness_message'] = "Please move closer"
                        return display_frame, results

                    # Calculate expected face size for current depth
                    depth_ratio = 500 / depth_mm
                    expected_size_ratio = 0.6 * depth_ratio
                    
                    if oval_guide:
                        actual_size_ratio = face_height / oval_guide['height']
                        size_deviation = abs(actual_size_ratio - expected_size_ratio)
                        
                        if size_deviation > 0.2:
                            logging.warning(f"Face size inconsistent with depth. Expected ratio: {expected_size_ratio:.2f}, Actual: {actual_size_ratio:.2f}")
                            results['liveness_message'] = "Please adjust your position"
                            return display_frame, results
                        
                        results['depth_valid'] = True

            # Update face position
            results['face_detected'] = True
            results['face_position'] = {
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'center_x': (x1 + x2) // 2,
                'center_y': (y1 + y2) // 2,
                'width': face_width,
                'height': face_height,
                'aspect_ratio': face_aspect_ratio
            }

            if depth_mm is not None:
                results['depth_mm'] = depth_mm
                results['depth_confidence'] = depth_confidence

            # Check oval guide constraints with improved validation
            if oval_guide:
                oval_center_x = oval_guide['x'] + oval_guide['width'] // 2
                oval_center_y = oval_guide['y'] + oval_guide['height'] // 2
                face_center_x = (x1 + x2) // 2
                face_center_y = (y1 + y2) // 2
                
                # Calculate relative distances and sizes
                rel_distance_x = abs(face_center_x - oval_center_x) / oval_guide['width']
                rel_distance_y = abs(face_center_y - oval_center_y) / oval_guide['height']
                face_width_ratio = face_width / oval_guide['width']
                face_height_ratio = face_height / oval_guide['height']
                
                # More precise position validation
                if rel_distance_x > 0.2 or rel_distance_y > 0.2:  # Reduced from 0.5 to 0.2
                    results['liveness_message'] = "Center your face in the oval"
                    return display_frame, results
                
                # Size validation with depth correlation
                if depth_mm is not None:
                    depth_factor = 500 / depth_mm
                    expected_width_ratio = 0.7 * depth_factor  # Face should occupy ~70% of oval width at optimal depth
                    expected_height_ratio = 0.8 * depth_factor # Face should occupy ~80% of oval height at optimal depth
                    
                    width_deviation = abs(face_width_ratio - expected_width_ratio)
                    height_deviation = abs(face_height_ratio - expected_height_ratio)
                    
                    if width_deviation > 0.15 or height_deviation > 0.15:
                        if face_width_ratio < expected_width_ratio - 0.15:
                            results['liveness_message'] = "Please move closer"
                        elif face_width_ratio > expected_width_ratio + 0.15:
                            results['liveness_message'] = "Please move back"
                        return display_frame, results

                # Run anti-spoofing check
                face_img = frame[y1:y2, x1:x2]
                if face_img is not None and face_img.size > 0:
                    pred = self.anti_spoofing.check_liveness(
                        frame, (x1, y1, x2, y2), depth_mm, depth_confidence, 
                        landmarks=points, oval_guide=oval_guide
                    )
                else:
                    return display_frame, results

                is_live, message, confidence = pred
                logging.info(f"Liveness check result: is_live={is_live}, confidence={confidence:.2f}, message='{message}'")
                
                results['is_live'] = is_live
                results['liveness_confidence'] = float(confidence)
                results['liveness_message'] = message
                
                if is_live:
                    if confidence > 0.5:
                        logging.info("Running face recognition")
                        name, recognition_confidence = self.face_recognition.recognize_face(frame)
                        logging.info(f"Recognition result: name='{name}', confidence={recognition_confidence:.2f}")
                        results['recognized_name'] = name
                        results['recognition_confidence'] = recognition_confidence
                        if name and name != "Unknown":
                            results['liveness_message'] = "Face verified successfully"
                else:
                    logging.warning("Liveness check failed")
                    results['liveness_message'] = message or "Please look directly at the camera"
            else:
                logging.error("Failed to extract face image for anti-spoofing check")

            # Calculate processing time
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            logging.info(f"Frame processing completed in {processing_time:.3f} seconds")
            
            return display_frame, results
            
        except Exception as e:
            logging.error(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame, {'success': False, 'error': str(e)}

    def get_processing_stats(self) -> dict:
        """Return current processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'faces_detected': 0,
            'successful_alignments': 0,
            'depth_measurements': 0,
            'spoof_attempts': 0,
            'avg_fps': 0,
            'avg_processing_time': 0
        }