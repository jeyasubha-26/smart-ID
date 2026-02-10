"""
Face Recognition Module for Smart ID - Face Recognition Based Attendance System

This module handles:
1. Loading face images from dataset_reference
2. Generating face embeddings using DeepFace (FaceNet model)
3. Storing embeddings for future reuse in embeddings/ folder
4. Loading test images from dataset_test
5. Comparing test embeddings with reference embeddings
6. Identifying the best matching student with confidence scores
"""

import os
import pickle
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from deepface import DeepFace
import logging
from dataset_loader import DatasetLoader

# Suppress TensorFlow and DeepFace logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN logs
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'  # Suppress deprecation warnings

# Configure logging with force=True to prevent duplicate handlers
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class FaceRecognition:
    """
    Handles face recognition using DeepFace (FaceNet) for attendance system.
    """
    
    def __init__(self, model_name: str = 'Facenet', similarity_threshold: float = 0.65):
        """
        Initialize the face recognition system.
        
        Args:
            model_name: DeepFace model to use (Facenet, VGG-Face, etc.)
            similarity_threshold: Minimum similarity score for positive recognition
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.base_dir = Path(__file__).parent.parent
        self.embeddings_dir = self.base_dir / "embeddings"
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader()
        
        # Create embeddings directory
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for embeddings
        self.reference_embeddings = {}
        
        logger.info(f"Face recognition initialized with model: {model_name}, threshold: {similarity_threshold}")
    
    def generate_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Generate and store embeddings for all reference images.
        
        Returns:
            Dictionary mapping person names to their average embedding vectors
        """
        try:
            logger.info("Generating reference embeddings...")
            
            # Load reference images
            reference_images = self.dataset_loader.get_reference_images()
            
            if not reference_images:
                logger.warning("No reference images found!")
                return {}
            
            person_embeddings = {}
            
            for person_name, image_paths in reference_images.items():
                logger.info(f"Processing {person_name}: {len(image_paths)} images")
                embeddings = []
                
                for image_path in image_paths:
                    try:
                        # Generate embedding using DeepFace
                        embedding = self._generate_single_embedding(image_path)
                        if embedding is not None:
                            embeddings.append(embedding)
                    except Exception as e:
                        logger.warning(f"Failed to process {image_path}: {e}")
                
                # Calculate average embedding for the person
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    # Apply L2 normalization to average embedding
                    norm = np.linalg.norm(avg_embedding)
                    if norm > 0:
                        avg_embedding = avg_embedding / norm
                        logger.info(f"Applied L2 normalization for {person_name}")
                    person_embeddings[person_name] = avg_embedding
                    logger.info(f"Generated {len(embeddings)} embeddings for {person_name}")
                else:
                    logger.warning(f"No valid embeddings generated for {person_name}")
            
            # Store embeddings for future use
            self._save_embeddings(person_embeddings)
            self.reference_embeddings = person_embeddings
            
            logger.info(f"Reference embeddings generated for {len(person_embeddings)} people")
            return person_embeddings
            
        except Exception as e:
            logger.error(f"Error generating reference embeddings: {e}")
            raise
    
    def _generate_single_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single image using DeepFace.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding vector or None if face detection fails
        """
        try:
            # Use DeepFace to generate embedding
            embedding_obj = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection= False,
                detector_backend='opencv'
            )
            if embedding_obj and len(embedding_obj) > 0:
                # Extract the embedding vector
                embedding = embedding_obj[0]['embedding']
                return np.array(embedding)
            else:
                logger.warning(f"No face detected in {image_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Error generating embedding for {image_path}: {e}")
            return None
    
    def _save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """
        Save embeddings to pickle file for future reuse.
        
        Args:
            embeddings: Dictionary of person embeddings
        """
        try:
            embeddings_file = self.embeddings_dir / "reference_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Embeddings saved to {embeddings_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load previously saved reference embeddings.
        
        Returns:
            Dictionary mapping person names to their embedding vectors
        """
        try:
            embeddings_file = self.embeddings_dir / "reference_embeddings.pkl"
            
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.reference_embeddings = embeddings
                logger.info(f"Loaded {len(embeddings)} reference embeddings")
                return embeddings
            else:
                logger.info("No saved embeddings found, generating new ones...")
                return self.generate_reference_embeddings()
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            logger.info("Generating new embeddings...")
            return self.generate_reference_embeddings()
    
    def recognize_faces_from_frame(self, frame: np.ndarray) -> List[Tuple[str, float]]:
        """
        Recognize multiple faces from a webcam frame.
        
        Args:
            frame: OpenCV BGR image (numpy array)
            
        Returns:
            List of (predicted_name, confidence) tuples for each detected face
            Returns empty list if no faces are confidently recognized
        """
        try:
            # Ensure reference embeddings are loaded
            if not self.reference_embeddings:
                self.reference_embeddings = self.load_reference_embeddings()
            
            if not self.reference_embeddings:
                logger.error("No reference embeddings available!")
                return []
            
            # Detect faces and generate embeddings using DeepFace
            try:
                # Use DeepFace to detect faces and generate embeddings
                face_objs = DeepFace.represent(
                    img_path=frame,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                if not face_objs:
                    logger.debug("No faces detected in frame")
                    return []
                
                logger.info(f"Detected {len(face_objs)} faces in frame")
                
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")
                return []
            
            recognized_faces = []
            
            # Process each detected face
            for face_obj in face_objs:
                try:
                    # Extract embedding for this face
                    face_embedding = np.array(face_obj['embedding'])
                    
                    # Apply L2 normalization
                    norm = np.linalg.norm(face_embedding)
                    if norm > 0:
                        face_embedding = face_embedding / norm
                    
                    # Find best match among all reference embeddings
                    best_match = None
                    best_score = 0.0
                    
                    for person_name, ref_embedding in self.reference_embeddings.items():
                        # Calculate cosine similarity
                        similarity = self._calculate_cosine_similarity(face_embedding, ref_embedding)
                        
                        # Always find the BEST match (no early break)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = person_name
                    
                    # Apply strict threshold for recognition
                    if best_score >= self.similarity_threshold:
                        recognized_faces.append((best_match, best_score))
                        logger.info(f"Recognized {best_match} with confidence: {best_score:.3f}")
                    else:
                        logger.debug(f"Face detected but below threshold: {best_score:.3f} < {self.similarity_threshold}")
                
                except Exception as e:
                    logger.warning(f"Error processing detected face: {e}")
                    continue
            
            logger.info(f"Successfully recognized {len(recognized_faces)} faces from frame")
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Error in recognize_faces_from_frame: {e}")
            return []
    
    def recognize_face(self, test_image_path: str, 
                      reference_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Optional[str], float]:
        """
        Recognize a face from test image (legacy method for batch processing).
        
        Args:
            test_image_path: Path to test image
            reference_embeddings: Pre-loaded reference embeddings (optional)
            
        Returns:
            Tuple of (predicted_person_name, similarity_score)
        """
        try:
            # Use provided embeddings or load from storage
            if reference_embeddings is None:
                reference_embeddings = self.reference_embeddings
                if not reference_embeddings:
                    reference_embeddings = self.load_reference_embeddings()
            
            if not reference_embeddings:
                logger.error("No reference embeddings available!")
                return None, 0.0
            
            # Generate embedding for test image
            test_embedding = self._generate_single_embedding(test_image_path)
            if test_embedding is None:
                logger.warning(f"No face detected in test image: {test_image_path}")
                return None, 0.0
            
            # Apply L2 normalization to test embedding
            norm = np.linalg.norm(test_embedding)
            if norm > 0:
                test_embedding = test_embedding / norm
            
            # Compare with all reference embeddings
            best_match = None
            best_score = 0.0
            
            for person_name, ref_embedding in reference_embeddings.items():
                # Calculate cosine similarity (both embeddings are normalized)
                similarity = self._calculate_cosine_similarity(test_embedding, ref_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = person_name
            
            # Check if similarity meets threshold
            if best_score >= self.similarity_threshold:
                logger.debug(f"Recognized: {best_match} with similarity: {best_score:.3f}")
                return best_match, best_score
            else:
                logger.debug(f"No match found (best: {best_match} with {best_score:.3f} < {self.similarity_threshold})")
                return "UNKNOWN", best_score
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, 0.0
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        Both embeddings should be normalized for stable computation.
        
        Args:
            embedding1: First embedding vector (should be normalized)
            embedding2: Second embedding vector (should be normalized)
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # For normalized vectors, cosine similarity is just the dot product
            similarity = np.dot(embedding1, embedding2)
            # Ensure similarity is in 0-1 range (cosine similarity should be between -1 and 1)
            similarity = max(0.0, min(1.0, similarity))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
            
    def batch_recognize(self, test_images: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Perform batch recognition on test images.
        
        Args:
            test_images: Dictionary of test images (person_name -> list of image paths)
            
        Returns:
            Dictionary of recognition results (person_name -> list of (image_path, predicted_name, confidence))
        """
        try:
            if not test_images:
                test_images = self.dataset_loader.get_test_images()
            
            if not test_images:
                logger.warning("No test images found!")
                return {}
            
            # Ensure reference embeddings are loaded
            if not self.reference_embeddings:
                self.load_reference_embeddings()
            
            results = {}
            recognized_students = set()  # Prevent multiple attendance marking
            
            for person_name, image_paths in test_images.items():
                logger.info(f"Processing test images for {person_name}: {len(image_paths)} images")
                person_results = []
                
                try:
                    for image_path in image_paths:
                        predicted_name, confidence = self.recognize_face(image_path)
                        
                        # Only add to results if this student hasn't been recognized yet
                        if predicted_name and predicted_name not in recognized_students:
                            recognized_students.add(predicted_name)
                            person_results.append((image_path, predicted_name, confidence))
                        elif predicted_name:
                            # Skip already recognized student but log for debugging
                            logger.debug(f"Skipping already recognized student: {predicted_name}")
                        else:
                            # Always add failed recognitions for accuracy tracking
                            person_results.append((image_path, None, confidence))
                    
                    results[person_name] = person_results
                    
                except KeyboardInterrupt:
                    logger.info("Batch recognition stopped by user")
                    logger.info(f"Partial results for {len(results)} people processed")
                    return results
                except Exception as e:
                    logger.error(f"Error processing {person_name}: {e}")
                    continue
            
            logger.info(f"Batch recognition completed. Recognized {len(recognized_students)} unique students")
            return results
            
        except KeyboardInterrupt:
            logger.info("Batch recognition stopped by user")
            return {}
        except Exception as e:
            logger.error(f"Error in batch recognition: {e}")
            raise
    
    def calculate_accuracy(self, batch_results: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, float]:
        """
        Calculate recognition accuracy from batch results.
        
        Args:
            batch_results: Results from batch_recognize method
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            total_images = 0
            correct_predictions = 0
            unknown_predictions = 0
            
            for true_person, results in batch_results.items():
                for image_path, predicted_person, confidence in results:
                    total_images += 1
                    
                    if predicted_person is None:
                        unknown_predictions += 1
                    elif predicted_person == true_person:
                        correct_predictions += 1
            
            accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
            unknown_rate = (unknown_predictions / total_images) * 100 if total_images > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'total_images': total_images,
                'correct_predictions': correct_predictions,
                'unknown_predictions': unknown_predictions,
                'unknown_rate': unknown_rate
            }
            
            logger.info(f"Recognition Accuracy: {accuracy:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return {}


def main():
    """
    Main function to test the face recognition system.
    """
    try:
        # Initialize face recognition system
        face_rec = FaceRecognition(model_name='Facenet', similarity_threshold=0.65)
        
        # Generate or load reference embeddings
        logger.info("Loading reference embeddings...")
        reference_embeddings = face_rec.load_reference_embeddings()
        
        # Perform batch recognition on test images
        logger.info("Starting batch recognition...")
        batch_results = face_rec.batch_recognize()
        
        # Calculate and display accuracy
        if batch_results:
            accuracy_metrics = face_rec.calculate_accuracy(batch_results)
            
            logger.info("=== Recognition Results ===")
            for person, results in batch_results.items():
                logger.info(f"\n{person}:")
                for image_path, predicted, confidence in results[:3]:  # Show first 3 results
                    status = "✓" if predicted == person else "✗" if predicted else "?"
                    logger.info(f"  {status} {predicted or 'Unknown'} ({confidence:.3f})")
            
            logger.info("\n=== Accuracy Metrics ===")
            for key, value in accuracy_metrics.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("Face recognition test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()