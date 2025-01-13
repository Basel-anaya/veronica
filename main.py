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


# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.recognition_threshold = 0.35  # Lower threshold for InsightFace
        
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
                face_input = cv2.cvtColor(face_input, cv2.COLOR_RGB2BGR)
            
            # Log shape for debugging
            logging.info(f"Face input shape: {face_input.shape}")
            
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
            
            # Log embedding shape for debugging
            logging.info(f"Extracted embedding shape: {embedding.shape}")
            
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
            
            # Extract embeddings for the input face
            query_embeddings = self.extract_face_embeddings(face_image)
            if query_embeddings is None:
                logging.warning("Could not extract embeddings from input face")
                return None, 0.0
            
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
            
            # Log similarity scores for debugging
            logging.info(f"Best match similarity score: {max_similarity:.4f}")
            logging.info(f"Best match name: {self.names_list[max_similarity_idx]}")
            logging.info(f"Recognition threshold: {self.recognition_threshold}")
            
            if max_similarity > self.recognition_threshold:
                return self.names_list[max_similarity_idx], float(max_similarity)
            else:
                logging.info(f"Similarity {max_similarity:.4f} below threshold {self.recognition_threshold}")
                return "Unknown", float(max_similarity)
                
        except Exception as e:
            logging.error(f"Error in recognize_face: {str(e)}")
            traceback.print_exc()
            return "Error", 0.0

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
        self.max_history = 10
        self.last_valid_depth = None
        
        # Motion analysis parameters
        self.last_landmarks = None
        self.last_depth = None
        self.motion_history = []
        self.max_motion_history = 10  # Increased history for better micro-movement analysis
        self.micro_movement_threshold = 0.001  # Threshold for natural micro-movements
        self.max_movement = 0.1  # Maximum expected natural movement
        self.landmark_groups = {
            'left_eye': list(range(33, 246)),    # Left eye region
            'right_eye': list(range(362, 466)),  # Right eye region
            'mouth': list(range(0, 17)),         # Jaw line
            'nose': list(range(168, 197)),       # Nose region
        }
        self.differential_weights = {
            'left_eye': 1.5,    # More weight to eye movements
            'right_eye': 1.5,   # More weight to eye movements
            'mouth': 1.0,       # Normal weight to mouth
            'nose': 0.8,        # Less weight to nose (should be more stable)
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
            
            # Apply temporal smoothing
            self.depth_history.append(depth_mm)
            if len(self.depth_history) > self.max_history:
                self.depth_history.pop(0)
            
                # Use median for final depth
            smoothed_depth = np.median(self.depth_history)
            
            # Calculate final confidence
            temporal_confidence = 1.0 / (1.0 + np.std(self.depth_history) / 10.0)
            final_confidence = measurement_confidence * temporal_confidence
                
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

    def _analyze_micro_movements(self, landmarks: np.ndarray) -> Tuple[float, dict]:
        """Analyze micro-movements in different facial regions."""
        movements = {}
        
        try:
            if self.last_landmarks is None:
                return 0.0, {}
            
            # Calculate movements for each facial region
            for region, indices in self.landmark_groups.items():
                current_points = landmarks[indices]
                previous_points = self.last_landmarks[indices]
                
                # Calculate differential movement
                movement = np.mean(np.abs(current_points - previous_points))
                
                # Apply regional weights
                weighted_movement = movement * self.differential_weights[region]
                movements[region] = weighted_movement
            
            # Calculate overall movement score
            total_movement = sum(movements.values()) / len(movements)
            
            return total_movement, movements
            
        except Exception as e:
            print(f"Error in micro-movement analysis: {str(e)}")
            return 0.0, {}

    def _check_motion_parallax(self, landmarks: np.ndarray, depth_mm: float) -> Tuple[bool, float]:
        """Check for natural facial micro-movements considering depth."""
        try:
            if self.last_landmarks is None:
                self.last_landmarks = landmarks.copy()
                self.last_depth = depth_mm
                return True, 0.8  # Initial confidence
            
            # Analyze micro-movements in different facial regions
            total_movement, regional_movements = self._analyze_micro_movements(landmarks)
            
            # Add to history
            self.motion_history.append(total_movement)
            if len(self.motion_history) > self.max_motion_history:
                self.motion_history.pop(0)
            
            # Calculate weighted average of recent movements
            weights = np.linspace(0.7, 1.0, len(self.motion_history))
            weights = weights / np.sum(weights)
            avg_movement = np.average(self.motion_history, weights=weights)
            
            # Adjust thresholds based on depth
            depth_cm = depth_mm / 10.0
            depth_factor = 50.0 / depth_cm  # Normalize to 50cm
            adjusted_micro = self.micro_movement_threshold * depth_factor
            adjusted_max = self.max_movement * depth_factor
            
            # Check if movement is within natural range
            is_natural = adjusted_micro <= avg_movement <= adjusted_max
            
            # Calculate confidence
            if avg_movement < adjusted_micro:
                confidence = 0.5 + (0.5 * (avg_movement / adjusted_micro))
            elif avg_movement > adjusted_max:
                confidence = 0.5 + (0.5 * (adjusted_max / avg_movement))
            else:
                # Higher confidence in the natural range
                optimal_movement = (adjusted_micro + adjusted_max) / 2
                deviation = abs(avg_movement - optimal_movement)
                max_deviation = (adjusted_max - adjusted_micro) / 2
                confidence = 0.7 + (0.3 * (1.0 - (deviation / max_deviation)))
            
            # Print debug info for movement analysis
            print(f"Movement Analysis:")
            print(f"  Depth: {depth_cm:.1f}cm")
            print(f"  Total Movement: {avg_movement:.6f}")
            print(f"  Adjusted Thresholds: {adjusted_micro:.6f} - {adjusted_max:.6f}")
            for region, movement in regional_movements.items():
                print(f"  {region}: {movement:.6f}")
            print(f"  Natural: {is_natural}, Confidence: {confidence:.3f}")
            
            # Update last values
            self.last_landmarks = landmarks.copy()
            self.last_depth = depth_mm
            
            return is_natural, float(confidence)
            
        except Exception as e:
            print(f"Error in motion analysis: {str(e)}")
            return True, 0.6


class ReflectionDetector:
    def __init__(self):
        """Initialize reflection detection parameters."""
        # Thresholds for reflection detection
        self.highlight_threshold = 220  # Threshold for specular highlights
        self.min_highlight_area = 10    # Minimum area of highlights to consider
        self.max_highlight_area = 1000  # Maximum area of highlights to consider
        self.reflection_history = []
        self.max_history = 10

    def detect_reflections(self, frame: np.ndarray, face_region: tuple) -> Tuple[float, list]:
        """
        Detect screen reflections and specular highlights.
        Returns confidence score and list of reflection points.
        """
        try:
            x1, y1, x2, y2 = face_region
            face_img = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Extract highlights using adaptive thresholding
            _, highlights = cv2.threshold(gray, self.highlight_threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours of highlights
            contours, _ = cv2.findContours(highlights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and analyze highlight patterns
            valid_highlights = []
            screen_like_pattern = False
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_highlight_area < area < self.max_highlight_area:
                    # Get bounding rect and aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h if h > 0 else 0
                    
                    # Check if highlight pattern resembles screen reflection
                    if 0.8 < aspect_ratio < 1.2:  # Nearly square highlights typical of screens
                        screen_like_pattern = True
                    
                    # Add to valid highlights
                    valid_highlights.append({
                        'position': (x+x1, y+y1),  # Global coordinates
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
            
            # Calculate confidence score
            if len(valid_highlights) == 0:
                confidence = 0.8  # No suspicious reflections
            else:
                # Lower confidence if screen-like patterns are detected
                base_confidence = 0.3 if screen_like_pattern else 0.6
                # Adjust based on number of highlights
                confidence = base_confidence * (1.0 / (1.0 + len(valid_highlights) * 0.1))
            
            return confidence, valid_highlights
            
        except Exception as e:
            print(f"Error in reflection detection: {str(e)}")
            return 0.0, []

    def draw_debug(self, frame: np.ndarray, reflections: list) -> np.ndarray:
        """Draw detected reflections for visualization."""
        try:
            for highlight in reflections:
                pos = highlight['position']
                cv2.circle(frame, pos, 3, (0, 165, 255), -1)  # Orange dot
                
                # Draw confidence text
                cv2.putText(frame, f"A:{highlight['area']:.0f}", 
                           (pos[0]+5, pos[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           (0, 165, 255), 1)
            
            return frame
        except Exception as e:
            print(f"Error drawing reflections: {str(e)}")
            return frame


class AntiSpoofing:
    def __init__(self):
        """Initialize anti-spoofing system with multiple detection methods."""
        # Define depth ranges in cm
        self.DEPTH_CONSTRAINTS = {
            'min_distance_cm': 20.0,    # Minimum working distance
            'max_distance_cm': 100.0,   # Maximum working distance
            'optimal_distance_cm': 50.0, # Optimal distance for recognition
            'high_confidence_range': (35.0, 75.0),
            'medium_confidence_range': (25.0, 90.0),
        }
        
        # Head-frame ratio constraints based on depth
        self.HEAD_RATIO_CONSTRAINTS = {
            'optimal_ratio_at_50cm': 0.55,  # At 50cm, head should be ~55% of frame height
            'min_ratio_at_100cm': 0.35,     # At 100cm, head should be at least 35%
            'max_ratio_at_20cm': 0.85,      # At 20cm, head should be at most 85%
            'tolerance_factor': 0.35        # Allow 35% deviation from expected ratio
        }
        
        # Focus/blur detection parameters
        self.FOCUS_THRESHOLD = 80.0  # Adjusted for better sensitivity
        self.blur_history = []
        self.max_blur_history = 5
        
        # Motion analysis parameters
        self.last_landmarks = None
        self.last_depth = None
        self.motion_history = []
        self.max_motion_history = 4
        self.micro_movement_threshold = 0.001
        self.max_movement = 0.15
        self.landmark_groups = {
            'left_eye': list(range(33, 246)),
            'right_eye': list(range(362, 466)),
            'mouth': list(range(0, 17)),
            'nose': list(range(168, 197)),
        }
        self.differential_weights = {
            'left_eye': 1.5,
            'right_eye': 1.5,
            'mouth': 0.7,
            'nose': 1.2,
        }
        
        # Lens distortion analysis parameters
        self.distortion_history = []
        self.max_distortion_history = 5  # Reduced for faster response
        self.distortion_groups = {
            'left_eye': [33, 133, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 263, 384, 385, 386, 387, 388, 466],
            'mouth': [0, 17, 61, 291, 314, 405],
            'nose': [168, 197, 2, 98],
        }
        
        # Results history
        self.liveness_history = []
        self.max_liveness_history = 3

    def _analyze_micro_movements(self, landmarks: np.ndarray) -> Tuple[float, dict]:
        """Analyze micro-movements in different facial regions."""
        movements = {}
        
        try:
            if self.last_landmarks is None:
                return 0.0, {}
            
            # Calculate movements for each facial region
            for region, indices in self.landmark_groups.items():
                current_points = landmarks[indices]
                previous_points = self.last_landmarks[indices]
                
                # Calculate differential movement
                movement = np.mean(np.abs(current_points - previous_points))
                
                # Apply regional weights
                weighted_movement = movement * self.differential_weights[region]
                movements[region] = weighted_movement
            
            # Calculate overall movement score
            total_movement = sum(movements.values()) / len(movements)
            
            return total_movement, movements
            
        except Exception as e:
            print(f"Error in micro-movement analysis: {str(e)}")
            return 0.0, {}

    def _check_motion_parallax(self, landmarks: np.ndarray, depth_mm: float) -> Tuple[bool, float]:
        """Check for natural facial micro-movements considering depth."""
        try:
            if self.last_landmarks is None:
                self.last_landmarks = landmarks.copy()
                self.last_depth = depth_mm
                return True, 0.8  # Initial confidence
            
            # Analyze micro-movements in different facial regions
            total_movement, regional_movements = self._analyze_micro_movements(landmarks)
            
            # Add to history
            self.motion_history.append(total_movement)
            if len(self.motion_history) > self.max_motion_history:
                self.motion_history.pop(0)
            
            # Calculate weighted average of recent movements
            weights = np.linspace(0.7, 1.0, len(self.motion_history))
            weights = weights / np.sum(weights)
            avg_movement = np.average(self.motion_history, weights=weights)
            
            # Adjust thresholds based on depth
            depth_cm = depth_mm / 10.0
            depth_factor = 50.0 / depth_cm  # Normalize to 50cm
            adjusted_micro = self.micro_movement_threshold * depth_factor
            adjusted_max = self.max_movement * depth_factor
            
            # Check if movement is within natural range
            is_natural = adjusted_micro <= avg_movement <= adjusted_max
            
            # Calculate confidence
            if avg_movement < adjusted_micro:
                confidence = 0.5 + (0.5 * (avg_movement / adjusted_micro))
            elif avg_movement > adjusted_max:
                confidence = 0.5 + (0.5 * (adjusted_max / avg_movement))
            else:
                # Higher confidence in the natural range
                optimal_movement = (adjusted_micro + adjusted_max) / 2
                deviation = abs(avg_movement - optimal_movement)
                max_deviation = (adjusted_max - adjusted_micro) / 2
                confidence = 0.7 + (0.3 * (1.0 - (deviation / max_deviation)))
            
            # Print debug info for movement analysis
            print(f"Movement Analysis:")
            print(f"  Depth: {depth_cm:.1f}cm")
            print(f"  Total Movement: {avg_movement:.6f}")
            print(f"  Adjusted Thresholds: {adjusted_micro:.6f} - {adjusted_max:.6f}")
            for region, movement in regional_movements.items():
                print(f"  {region}: {movement:.6f}")
            print(f"  Natural: {is_natural}, Confidence: {confidence:.3f}")
            
            # Update last values
            self.last_landmarks = landmarks.copy()
            self.last_depth = depth_mm
            
            return is_natural, float(confidence)
            
        except Exception as e:
            print(f"Error in motion analysis: {str(e)}")
            return True, 0.6

    def _analyze_lens_distortion(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[bool, float]:
        """Analyze lens distortion patterns to detect real faces vs photos."""
        try:
            h, w = frame.shape[:2]
            frame_center = np.array([w/2, h/2])
            distortion_scores = []
            
            # Convert landmark indices to points
            points = landmarks
            
            for group_name, indices in self.distortion_groups.items():
                group_points = points[indices]
                
                # Calculate distances from frame center
                distances = np.linalg.norm(group_points - frame_center, axis=1)
                max_possible_distance = np.linalg.norm([w/2, h/2])
                relative_distances = distances / max_possible_distance
                
                # Calculate feature size ratios
                feature_sizes = []
                for i in range(len(indices)-1):
                    # Calculate size as area of triangle with frame center
                    p1 = group_points[i]
                    p2 = group_points[i+1]
                    v1 = p1 - frame_center
                    v2 = p2 - frame_center
                    area = abs(np.cross(v1, v2)) / 2
                    feature_sizes.append(area)
                
                if not feature_sizes:
                    continue
                    
                # Normalize sizes
                max_size = max(feature_sizes)
                if max_size > 0:
                    size_ratios = np.array(feature_sizes) / max_size
                else:
                    size_ratios = np.ones_like(feature_sizes)
                
                # Analyze distortion based on distance from center
                for dist, ratio in zip(relative_distances, size_ratios):
                    if dist < 0.33:  # Center region
                        expected_range = (0.8, 1.2)
                        weight = 1.0
                    elif dist < 0.66:  # Middle region
                        expected_range = (0.7, 1.3)
                        weight = 0.8
                    else:  # Edge region
                        expected_range = (0.6, 1.4)
                        weight = 0.6
                    
                    # Calculate score
                    if expected_range[0] <= ratio <= expected_range[1]:
                        score = 1.0
                    else:
                        deviation = min(abs(ratio - expected_range[0]),
                                     abs(ratio - expected_range[1]))
                        score = 1.0 / (1.0 + (deviation * 4)**2)
                    
                    score *= weight
                    distortion_scores.append(score)
            
            if not distortion_scores:
                return True, 0.5
            
            # Calculate overall distortion confidence
            scores = np.array(distortion_scores)
            weights = scores / np.sum(scores)
            avg_score = np.average(scores, weights=weights)
            
            # Add to history
            self.distortion_history.append(avg_score)
            if len(self.distortion_history) > self.max_distortion_history:
                self.distortion_history.pop(0)
            
            # Calculate weighted average of recent scores
            weights = np.linspace(0.6, 1.0, len(self.distortion_history))
            weights = weights / np.sum(weights)
            final_score = np.average(self.distortion_history, weights=weights)
            
            # Determine if natural
            is_natural = final_score > 0.6
            
            return is_natural, float(final_score)
            
        except Exception as e:
            print(f"Error in lens distortion analysis: {str(e)}")
            return True, 0.5

    def _check_focus(self, frame: np.ndarray, face_region: tuple) -> Tuple[bool, float]:
        """Check if face region is in focus using Laplacian variance."""
        try:
            x1, y1, x2, y2 = face_region
            face_img = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of focus)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_measure = laplacian.var()
            
            # Add to history
            self.blur_history.append(focus_measure)
            if len(self.blur_history) > self.max_blur_history:
                self.blur_history.pop(0)
            
            # Use median for stability
            focus_measure = np.median(self.blur_history)
            
            # Calculate confidence
            confidence = min(1.0, focus_measure / self.FOCUS_THRESHOLD)
            is_focused = focus_measure > self.FOCUS_THRESHOLD
            
            return is_focused, float(confidence)
            
        except Exception as e:
            print(f"Error in focus check: {str(e)}")
            return True, 0.5

    def _check_gaze_direction(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[bool, float]:
        """Check if eyes are looking directly at camera using iris landmarks."""
        try:
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Get eye corners from landmarks array
            left_eye_inner = landmarks[133]  # Left eye inner corner
            left_eye_outer = landmarks[33]   # Left eye outer corner
            right_eye_inner = landmarks[362] # Right eye inner corner
            right_eye_outer = landmarks[263] # Right eye outer corner
            
            # Get iris centers from landmarks array
            left_eye_top = landmarks[159]    # Left eye top
            left_eye_bottom = landmarks[145] # Left eye bottom
            right_eye_top = landmarks[386]   # Right eye top
            right_eye_bottom = landmarks[374] # Right eye bottom
            
            # Calculate iris centers from eye landmarks
            left_iris_center = (left_eye_top + left_eye_bottom) / 2
            right_iris_center = (right_eye_top + right_eye_bottom) / 2
            
            def calculate_relative_position(eye_inner, eye_outer, iris_center):
                eye_width = np.linalg.norm(eye_outer - eye_inner)
                if eye_width == 0:
                    return 0.5
                
                # Project iris position onto eye line
                eye_direction = eye_outer - eye_inner
                eye_direction = eye_direction / np.linalg.norm(eye_direction)
                iris_relative = iris_center - eye_inner
                projection = np.dot(iris_relative, eye_direction)
                
                # Normalize to get relative position
                return projection / eye_width
            
            # Calculate relative positions (0.5 means perfectly centered)
            left_pos = calculate_relative_position(left_eye_inner, left_eye_outer, left_iris_center)
            right_pos = calculate_relative_position(right_eye_inner, right_eye_outer, right_iris_center)
            
            # Average the positions with smoothing
            avg_pos = (left_pos + right_pos) / 2
            
            # Calculate deviation from center (0.5) with more tolerance
            deviation = abs(avg_pos - 0.5)
            
            # Add to history (with less history for faster response)
            self.gaze_history.append(deviation)
            if len(self.gaze_history) > self.max_gaze_history:
                self.gaze_history.pop(0)
            
            # Use median for more stable results
            median_deviation = np.median(self.gaze_history)
            
            # More lenient check
            is_centered = median_deviation < self.gaze_center_threshold
            confidence = 1.0 - (median_deviation / 0.8)  # Even more lenient confidence calculation
            confidence = max(0.3, min(1.0, confidence))  # Clamp between 0.3 and 1.0
            
            return is_centered, confidence
            
        except Exception as e:
            print(f"Error in gaze direction check: {str(e)}")
            return True, 0.5

    def check_liveness(self, frame: np.ndarray, face_region: tuple, depth_mm: float, 
                      confidence: float, landmarks: list = None) -> Tuple[bool, str, float]:
        """Check if the face is live using multiple detection methods."""
        try:
            depth_cm = depth_mm / 10.0
            
            # 1. Depth Check (20%)
            if depth_cm < self.DEPTH_CONSTRAINTS['min_distance_cm']:
                return False, f"Too close: {depth_cm:.1f}cm", 0.3
            if depth_cm > self.DEPTH_CONSTRAINTS['max_distance_cm']:
                return False, f"Too far: {depth_cm:.1f}cm", 0.3
            
            # Calculate depth confidence
            if (self.DEPTH_CONSTRAINTS['high_confidence_range'][0] <= depth_cm <= 
                self.DEPTH_CONSTRAINTS['high_confidence_range'][1]):
                depth_confidence = 0.9
            elif (self.DEPTH_CONSTRAINTS['medium_confidence_range'][0] <= depth_cm <= 
                  self.DEPTH_CONSTRAINTS['medium_confidence_range'][1]):
                depth_confidence = 0.7
            else:
                depth_confidence = 0.5
            
            # 2. Head Size Check (20%)
            head_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            frame_height = frame.shape[0]
            head_ratio = head_height / frame_height
            
            expected_ratio = self.HEAD_RATIO_CONSTRAINTS['optimal_ratio_at_50cm'] * (50.0 / depth_cm)
            ratio_tolerance = self.HEAD_RATIO_CONSTRAINTS['tolerance_factor'] * expected_ratio
            is_valid_head_size = abs(head_ratio - expected_ratio) <= ratio_tolerance
            
            if not is_valid_head_size:
                return False, "Invalid head size for distance", 0.3
            
            head_size_confidence = 1.0 - (abs(head_ratio - expected_ratio) / ratio_tolerance)
            head_size_confidence = max(0.0, min(1.0, head_size_confidence))
            
            # 3. Motion Analysis (25%)
            is_natural_motion, motion_confidence = self._check_motion_parallax(landmarks, depth_mm)
            if not is_natural_motion:
                return False, "Unnatural movement detected", 0.3
            
            # 4. Lens Distortion Check (20%)
            is_natural_distortion, distortion_confidence = self._analyze_lens_distortion(frame, landmarks)
            if not is_natural_distortion:
                return False, "Unnatural lens distortion", 0.3
            
            # 5. Focus Check (15%)
            is_focused, focus_confidence = self._check_focus(frame, face_region)
            if not is_focused:
                return False, "Image out of focus", 0.3
            
            # Combine all confidence scores with weights
            final_confidence = (
                0.20 * depth_confidence +       # Depth (20%)
                0.20 * head_size_confidence +   # Head Size (20%)
                0.25 * motion_confidence +      # Motion (25%)
                0.20 * distortion_confidence +  # Distortion (20%)
                0.15 * focus_confidence         # Focus (15%)
            ) * confidence  # Multiply by input confidence
            
            # Make liveness decision
            is_live = final_confidence > 0.5
            
            # Update history
            self.liveness_history.append(is_live)
            if len(self.liveness_history) > self.max_liveness_history:
                self.liveness_history.pop(0)
            
            # Generate detailed message
            if is_live:
                message = f"Real face at {depth_cm:.1f}cm"
            else:
                if motion_confidence < 0.5:
                    message = "Unnatural movement"
                elif distortion_confidence < 0.5:
                    message = "Suspicious lens pattern"
                elif focus_confidence < 0.5:
                    message = "Poor image quality"
                elif head_size_confidence < 0.5:
                    message = "Invalid head size"
                else:
                    message = "Multiple checks failed"
            
            return is_live, message, final_confidence
            
        except Exception as e:
            print(f"Error in liveness check: {str(e)}")
            return True, "System check", 0.5

    def draw_feedback(self, frame: np.ndarray, is_live: bool, message: str, 
                     confidence: float, face_region: tuple, depth_mm: float) -> np.ndarray:
        """Draw anti-spoofing feedback on the frame."""
        try:
            # Draw basic feedback
            height, width = frame.shape[:2]
            font_scale = height / 720.0
            thickness = max(1, int(height / 500))
            
            # Define colors
            success_color = (0, 255, 0)  # Green for positive feedback
            warning_color = (0, 165, 255)  # Orange for warnings
            error_color = (0, 0, 255)  # Red for errors
            
            # Choose color based on message
            if "look directly at camera" in message:
                color = warning_color
            else:
                color = success_color if is_live else error_color
            
            # Draw face rectangle
            x1, y1, x2, y2 = face_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw message at the top
            cv2.putText(frame, message,
                       (10, int(30 * font_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       color, thickness)
            
            # Draw status info at the bottom
            depth_cm = depth_mm / 10.0
            y_pos = height - int(90 * font_scale)
            cv2.putText(frame, f"Depth: {depth_cm:.1f}cm",
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       color, thickness)
            
            y_pos += int(30 * font_scale)
            status_text = "Status: REAL" if is_live else "Status: SPOOF"
            cv2.putText(frame, status_text,
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       color, thickness)
            
            y_pos += int(30 * font_scale)
            cv2.putText(frame, f"Confidence: {confidence:.2f}",
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       color, thickness)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing feedback: {str(e)}")
            return frame


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

    def process_frame(self, frame: np.ndarray, draw_debug: bool = True) -> Tuple[np.ndarray, dict]:
        """Process a single frame through the entire face processing pipeline."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame, {}

        try:
            start_time = time.time()
            
            # Skip frames if we're falling behind
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                self.skip_frames += 1
                return frame, {}
            self.last_frame_time = current_time

            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)

            # Initialize result dictionary
            results = {
                'faces': [],
                'depth_mm': None,
                'confidence': None,
                'aligned_faces': [],
                'processing_time': 0,
                'is_live': False,
                'liveness_confidence': 0.0,
                'liveness_message': "",
                'recognized_name': None,
                'new_exposure': None  # Add field for exposure changes
            }

            # Create visualization frame only if needed
            display_frame = frame.copy() if draw_debug else frame

            # Use depth estimator's face detection result
            face_landmarks = self.depth_estimator.get_iris_landmarks(processed_frame)
            if face_landmarks:
                self.processing_stats['faces_detected'] += 1
                
                # Get all facial landmarks for better face region calculation
                if self.depth_estimator.last_detection_result and self.depth_estimator.last_detection_result.face_landmarks:
                    landmarks = self.depth_estimator.last_detection_result.face_landmarks[0]
                    h, w = frame.shape[:2]
                    
                    # Get all landmark points
                    points = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] 
                                     for idx in range(468)])
                    
                    # Calculate face region from all landmarks
                    x1, y1 = np.min(points, axis=0).astype(int)
                    x2, y2 = np.max(points, axis=0).astype(int)
                    
                    # Add margin to face region (40% on each side)
                    margin_w = int((x2 - x1) * 0.4)
                    margin_h = int((y2 - y1) * 0.4)
                    x1 = max(0, x1 - margin_w)
                    y1 = max(0, y1 - margin_h)
                    x2 = min(frame.shape[1], x2 + margin_w)
                    y2 = min(frame.shape[0], y2 + margin_h)
                    
                    # Extract face region with padding if needed
                    face_region = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
                    face_region[:] = 128  # Gray padding
                    valid_h = min(frame.shape[0] - y1, y2-y1)
                    valid_w = min(frame.shape[1] - x1, x2-x1)
                    face_region[:valid_h, :valid_w] = processed_frame[y1:y1+valid_h, x1:x1+valid_w]
                    
                    if face_region.size > 0:
                        # Depth Estimation (prioritize this over alignment)
                        if self.process_depth:
                            depth_info = self.depth_estimator.estimate_depth(processed_frame)
                            if depth_info is not None:
                                depth_mm, depth_confidence = depth_info
                                results['depth_mm'] = depth_mm
                                results['confidence'] = depth_confidence
                                
                                # Anti-spoofing check with current exposure value
                                is_live, message, confidence = self.anti_spoofing.check_liveness(
                                    processed_frame, (x1, y1, x2, y2), depth_mm, depth_confidence, 
                                    landmarks=points
                                )
                                
                                results['is_live'] = is_live
                                results['liveness_confidence'] = confidence
                                results['liveness_message'] = message
                                
                                # Only proceed with recognition if face is live
                                if is_live and self.face_recognition and confidence > 0.5:
                                    # Extract face region for recognition
                                    face_img = processed_frame[y1:y2, x1:x2]
                                    if face_img.size > 0:
                                        # Face recognition
                                        name = self.face_recognition.recognize_face(face_img)
                                        results['recognized_name'] = name
                                
                                # Draw feedback on frame if requested
                                if draw_debug:
                                    self.anti_spoofing.draw_feedback(
                                        display_frame, is_live, message, confidence,
                                        (x1, y1, x2, y2), depth_mm
                                    )
                                    
                                    # Draw recognition result if available
                                    if results['recognized_name']:
                                        cv2.putText(
                                            display_frame,
                                            str(results['recognized_name']),
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 255, 0) if is_live else (0, 0, 255),
                                            2
                                        )

            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            
            return display_frame, results
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            traceback.print_exc()
            return frame, {}

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

    def register_face_from_id_card(self, id_card_image: np.ndarray, person_name: str) -> bool:
        """Register a new face from an ID card image using multiple variations."""
        try:
            if id_card_image is None or id_card_image.size == 0:
                logging.error("Invalid ID card image")
                return False
            
            logging.info(f"Processing ID card image for {person_name}, shape: {id_card_image.shape}")
            
            # Convert to RGB (required by InsightFace)
            rgb_image = cv2.cvtColor(id_card_image, cv2.COLOR_BGR2RGB)
            logging.info("Converted image to RGB")
            
            # Detect faces using InsightFace
            logging.info("Attempting face detection with InsightFace...")
            faces = self.face_recognition.app.get(rgb_image)
            logging.info(f"Found {len(faces)} faces in ID card")
            
            if not faces:
                logging.error("No face detected in ID card")
                return False
            
            # Get the largest face (most likely the main face in the ID)
            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            logging.info(f"Selected largest face with bbox: {largest_face.bbox}")
            
            # Extract face embedding
            face_embedding = largest_face.embedding
            if face_embedding is None:
                logging.error("Could not extract face embedding")
                return False
            
            logging.info("Successfully extracted face embedding")
            
            # Register face with multiple variations
            success = self.face_recognition.register_face_with_multiple_angles(rgb_image, person_name)
            logging.info(f"Registration with multiple angles result: {success}")
            
            if success:
                logging.info(f"Successfully registered {person_name} with multiple variations")
            else:
                logging.error(f"Failed to register {person_name}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error registering face from ID card: {str(e)}")
            traceback.print_exc()
            return False


def register_face():
    """Handle face registration from ID card images in the ID-samples folder."""
    try:
        print("Initializing Face System for registration...")
        face_system = FaceSystem(
            min_detection_confidence=0.5,
            target_fps=30,
            input_size=(640, 480)
        )
        
        # Get list of ID card images
        id_samples_dir = "ID-samples"
        if not os.path.exists(id_samples_dir):
            print(f"Error: {id_samples_dir} directory not found")
            return
            
        id_files = [f for f in os.listdir(id_samples_dir) if f.endswith('-ID.jpg')]
        if not id_files:
            print(f"No ID card images found in {id_samples_dir}")
            return
            
        print("\nFound ID card images:")
        for i, filename in enumerate(id_files, 1):
            print(f"{i}. {filename}")
        
        print("\nProcessing ID cards...")
        for filename in id_files:
            # Extract name from filename (remove '-ID.jpg')
            person_name = filename.replace('-ID.jpg', '')
            
            # Read ID card image
            image_path = os.path.join(id_samples_dir, filename)
            id_card_image = cv2.imread(image_path)
            
            if id_card_image is None:
                print(f"Error: Could not read image {filename}")
                continue
            
            print(f"\nProcessing {filename}...")
            print(f"Registering {person_name}...")
            
            if face_system.register_face_from_id_card(id_card_image, person_name):
                print(f"Successfully registered {person_name}")
            else:
                print(f"Failed to register {person_name}")
        
        print("\nRegistration process completed")
        
    except Exception as e:
        print(f"\nError in registration mode: {str(e)}")
        traceback.print_exc()
        cv2.destroyAllWindows()

def run_face_detection():
    """Run the face detection and recognition system using Tkinter interface."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    
    class FaceDetectionApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Face Detection System")
            
            # Initialize face system with lower resolution
            self.face_system = FaceSystem(
                min_detection_confidence=0.5,
                target_fps=15,
                input_size=(320, 240)
            )
            
            # Initialize webcam with proper error handling
            print("Opening webcam...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set camera exposure and shutter speed
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto exposure
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)      # Set exposure time (lower value = shorter exposure)
            
            # Create control panel frame
            self.control_frame = ttk.Frame(root, padding="5")
            self.control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
            
            # Create exposure control
            ttk.Label(self.control_frame, text="Exposure:").grid(row=0, column=0, padx=5)
            self.exposure_var = tk.IntVar(value=-5)
            self.exposure_slider = ttk.Scale(
                self.control_frame,
                from_=-8, to=0,
                variable=self.exposure_var,
                orient=tk.HORIZONTAL,
                command=self.update_camera_settings
            )
            self.exposure_slider.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
            
            # Create gain control
            ttk.Label(self.control_frame, text="Gain:").grid(row=1, column=0, padx=5)
            self.gain_var = tk.IntVar(value=0)
            self.gain_slider = ttk.Scale(
                self.control_frame,
                from_=0, to=100,
                variable=self.gain_var,
                orient=tk.HORIZONTAL,
                command=self.update_camera_settings
            )
            self.gain_slider.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
            
            # Create main frame
            self.main_frame = ttk.Frame(root, padding="10")
            self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Create video display label with fixed size
            self.video_label = ttk.Label(self.main_frame)
            self.video_label.grid(row=0, column=0, padx=5, pady=5)
            
            # Create stats label
            self.stats_label = ttk.Label(self.main_frame, justify=tk.LEFT, font=('TkDefaultFont', 10))
            self.stats_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.N)
            
            # Initialize counters
            self.frame_count = 0
            self.start_time = time.time()
            self.last_update = time.time()
            self.last_frame = None
            self.processing = False
            
            # Set up closing handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start frame updates
            self.update_frame()
        
        def update_camera_settings(self, *args):
            """Update camera exposure and gain settings."""
            try:
                # Update exposure
                exposure = self.exposure_var.get()
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                
                # Update gain
                gain = self.gain_var.get()
                self.cap.set(cv2.CAP_PROP_GAIN, gain)
                
            except Exception as e:
                print(f"Error updating camera settings: {str(e)}")
                
        def update_frame(self):
            """Update frame in Tkinter window."""
            if self.processing:
                self.root.after(10, self.update_frame)
                return
            
            try:
                self.processing = True
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.processing = False
                    self.root.after(10, self.update_frame)
                    return
                
                # Process frame
                processed_frame, results = self.face_system.process_frame(frame, draw_debug=True)
                
                if processed_frame is not None:
                    display_frame = processed_frame
                else:
                    display_frame = frame
                
                # Convert frame to PhotoImage
                image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image=image)
                
                # Update label
                self.video_label.imgtk = image
                self.video_label.configure(image=image)
                self.last_frame = image
                
                # Update stats
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                if elapsed_time >= 1.0:  # Update stats every second
                    fps = self.frame_count / elapsed_time
                    self.stats_label.config(text=(
                        f"FPS: {fps:.1f}\n"
                        f"Processing Time: {(elapsed_time/self.frame_count)*1000:.1f}ms\n"
                        f"Faces Detected: {self.face_system.processing_stats['faces_detected']}\n"
                        f"Depth Measurements: {self.face_system.processing_stats['depth_measurements']}"
                    ))
                    self.frame_count = 0
                    self.start_time = current_time
                
                self.last_update = current_time
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                traceback.print_exc()
            finally:
                self.processing = False
            
            # Schedule next update
            self.root.after(10, self.update_frame)
        
        def on_closing(self):
            """Handle window closing."""
            try:
                if self.cap is not None:
                    self.cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
            finally:
                self.root.destroy()
    
    try:
        print("Initializing Face Detection System...")
        root = tk.Tk()
        app = FaceDetectionApp(root)
        root.mainloop()
        
    except Exception as e:
        print(f"\nError in detection mode: {str(e)}")
        traceback.print_exc()
        if 'app' in locals() and hasattr(app, 'cap'):
            app.cap.release()
        if 'root' in locals():
            root.destroy()

def main():
    """Main function to choose between registration and detection modes."""
    while True:
        print("\nFace System Menu:")
        print("1. Register new face from ID card")
        print("2. Run face detection and recognition")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            register_face()
        elif choice == '2':
            run_face_detection()
        elif choice == '3':
            print("Exiting system...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
