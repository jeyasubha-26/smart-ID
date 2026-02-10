"""
Main Controller for Smart ID - Face Recognition Based Attendance System

This module acts as the main controller that:
1. Initializes all system components
2. Coordinates face recognition and attendance management
3. Provides clear console output of attendance results
4. Handles the complete attendance workflow
"""
import logging
from datetime import datetime
from typing import Dict, List, Set
from dataset_loader import DatasetLoader
from face_recognition import FaceRecognition
from attendance_manager import AttendanceManager
from email_notifier import EmailNotifier
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartIDController:
    """
    Main controller for the Smart ID attendance system.
    """
    
    def __init__(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing Smart ID Attendance System...")
            
            # Initialize core components
            self.dataset_loader = DatasetLoader()
            self.face_recognition = FaceRecognition()
            self.attendance_manager = AttendanceManager()
            self.email_notifier = EmailNotifier()
            
            # Session identifier for tracking
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info("Smart ID Attendance System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
    
    def prepare_dataset(self) -> bool:
        """
        Prepare the dataset by downloading and organizing images.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Preparing dataset...")
            
            # Check if reference images exist
            reference_images = self.dataset_loader.get_reference_images()
            
            if not reference_images:
                logger.info("No reference images found. Downloading LFW dataset...")
                self.dataset_loader.download_and_prepare_lfw_dataset()
                reference_images = self.dataset_loader.get_reference_images()
            
            if not reference_images:
                logger.error("Failed to prepare reference images")
                return False
            
            # Check if test images exist
            test_images = self.dataset_loader.get_test_images()
            
            if not test_images:
                logger.error("No test images found")
                return False
            
            logger.info(f"Dataset prepared: {len(reference_images)} people in reference, {len(test_images)} people in test")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return False
    
    def run_attendance_system(self) -> bool:
        """
        Run complete attendance system workflow with dataset-only recognition.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("SMART ID - FACE RECOGNITION ATTENDANCE SYSTEM")
            logger.info("=" * 60)
            
            # Step 1: Prepare dataset
            if not self.prepare_dataset():
                logger.error("Failed to prepare dataset")
                return False
            
            # Step 2: Generate and load reference embeddings
            logger.info("Loading reference embeddings...")
            
            # First, try to load existing embeddings to check count
            reference_embeddings = self.face_recognition.load_reference_embeddings()
            
            # If no embeddings exist or count is 0, generate new ones
            if not reference_embeddings or len(reference_embeddings) == 0:
                logger.info("Generating reference embeddings...")
                reference_embeddings = self.face_recognition.generate_reference_embeddings()
                
                if not reference_embeddings:
                    logger.error("Failed to generate reference embeddings")
                    return False
                
                logger.info(f"Reference embeddings generated for {len(reference_embeddings)} people")
                
                # Load newly generated embeddings
                reference_embeddings = self.face_recognition.load_reference_embeddings()
            
            if not reference_embeddings:
                logger.error("Failed to load reference embeddings")
                return False
            
            logger.info(f"Loaded {len(reference_embeddings)} reference embeddings")
            
            # Step 3: Perform dataset-based face recognition
            logger.info("Performing dataset-based face recognition...")
            attendance_data = self._run_dataset_attendance_session()
            
            if not attendance_data:
                logger.error("Dataset attendance session failed")
                return False
            
            # Step 4: Record attendance in database
            logger.info("Recording attendance in database...")
            self._record_attendance(attendance_data)
            
            # Step 5: Send email notification
            self._send_email_notification(attendance_data)
            
            # Step 6: Display attendance summary
            self._display_attendance_summary(attendance_data)
            
            logger.info("Attendance system completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running attendance system: {e}")
            return False
    
    def _run_dataset_attendance_session(self) -> Dict:
        """
        Run dataset-based attendance session using LFW test images.
        
        Returns:
            Dictionary with attendance analysis
        """
        try:
            logger.info("Starting dataset-based attendance session...")
            
            # Get all known students (only from reference images)
            all_students = set(self.dataset_loader.get_reference_images().keys())
            recognized_students = set()  # Track each student only once
            unknown_faces = []  # Track unknown faces for reporting
            
            # Get test images
            test_images = self.dataset_loader.get_test_images()
            
            if not test_images:
                logger.error("No test images found")
                return {}
            
            logger.info(f"Processing test images for {len(test_images)} people...")
            
            # Process each student's test images
            for true_person, image_paths in test_images.items():
                logger.info(f"Processing {true_person}: {len(image_paths)} images")
                
                student_recognitions = []  # Track all recognitions for this student
                
                # Process each test image for this student
                for image_path in image_paths:
                    try:
                        # Recognize face from test image
                        predicted_name, distance = self.face_recognition.recognize_face(image_path)
                        
                        if predicted_name:
                            student_recognitions.append((predicted_name, distance))
                        
                    except Exception as e:
                        logger.warning(f"Error processing {image_path}: {e}")
                        continue
                
                # Decide final identity using best distance (lowest = best match)
                if student_recognitions:
                    # Find recognition with lowest distance (best match)
                    best_match = min(student_recognitions, key=lambda x: x[1])
                    final_predicted_name, best_distance = best_match
                    
                    # Validate identity AND use realistic threshold
                    # Only mark present if predicted matches true person AND distance is reasonable
                    if (final_predicted_name == true_person and 
                        final_predicted_name not in recognized_students and 
                        best_distance <= 1.2):  # Realistic threshold
                        
                        recognized_students.add(final_predicted_name)
                        logger.info(f"Student marked present: {final_predicted_name} (distance: {best_distance:.3f})")
                    elif final_predicted_name != true_person:
                        logger.debug(f"Identity mismatch: {final_predicted_name} != {true_person} (distance: {best_distance:.3f})")
                    elif final_predicted_name in recognized_students:
                        logger.debug(f"Student already recognized: {final_predicted_name}")
                    elif best_distance > 1.2:
                        logger.debug(f"Distance too high: {best_distance:.3f} > 1.2 for {final_predicted_name}")
                else:
                    logger.debug(f"No recognitions found for {true_person}")
            
            # Determine absent students (only from registered students)
            absent_students = all_students - recognized_students
            
            # Prepare attendance details
            attendance_details = {}
            for student_name in recognized_students:
                attendance_details[student_name] = {
                    'status': 'present',
                    'confidence_scores': [0.8],  # Dummy confidence for compatibility
                    'test_images': ['dataset_image']  # Dummy image path for compatibility
                }
            
            # Prepare attendance data for database
            attendance_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'total_students': len(all_students),
                'present_students': len(recognized_students),
                'absent_students': len(absent_students),
                'attendance_rate': (len(recognized_students) / len(all_students)) * 100 if all_students else 0,
                'present_details': attendance_details,
                'absent_list': list(absent_students),
                'recognized_list': list(recognized_students),
                'unknown_faces': unknown_faces  # Add unknown faces tracking
            }
            
            logger.info(f"Dataset session complete: {len(recognized_students)} present, {len(absent_students)} absent")
            return attendance_data
            
        except Exception as e:
            logger.error(f"Error in dataset attendance session: {e}")
            return {}
    
    def _analyze_recognition_results(self, batch_results: Dict[str, List]) -> Dict:
        """
        Analyze face recognition results to determine attendance.
        
        Args:
            batch_results: Results from face recognition batch processing
            
        Returns:
            Dictionary with attendance analysis
        """
        try:
            logger.info("Analyzing recognition results...")
            
            # Get all known students
            all_students = set(self.dataset_loader.get_reference_images().keys())
            
            # Track recognized students
            recognized_students = set()
            attendance_details = {}
            
            for true_person, results in batch_results.items():
                for image_path, predicted_person, confidence in results:
                    if predicted_person and confidence >= self.face_recognition.similarity_threshold:
                        # Add to recognized set only once per person
                        recognized_students.add(predicted_person)
                        
                        # Initialize attendance details only once per person
                        if predicted_person not in attendance_details:
                            attendance_details[predicted_person] = {
                                'status': 'present',
                                'confidence_scores': [],
                                'test_images': []
                            }
                        
                        # Collect all confidence scores and image paths for analysis
                        attendance_details[predicted_person]['confidence_scores'].append(confidence)
                        attendance_details[predicted_person]['test_images'].append(image_path)
            
            # Determine absent students
            absent_students = all_students - recognized_students
            
            # Prepare attendance data for database
            attendance_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'total_students': len(all_students),
                'present_students': len(recognized_students),
                'absent_students': len(absent_students),
                'attendance_rate': (len(recognized_students) / len(all_students)) * 100 if all_students else 0,
                'present_details': attendance_details,
                'absent_list': list(absent_students),
                'recognized_list': list(recognized_students)
            }
            
            logger.info(f"Analysis complete: {len(recognized_students)} present, {len(absent_students)} absent")
            return attendance_data
            
        except Exception as e:
            logger.error(f"Error analyzing recognition results: {e}")
            return {}
    
    def _record_attendance(self, attendance_data: Dict):
        """
        Record attendance data in the database.
        
        Args:
            attendance_data: Analyzed attendance data
        """
        try:
            if not attendance_data:
                logger.error("No attendance data to record")
                return
            
            # Record present students
            batch_attendance_data = []
            
            for student_name, details in attendance_data['present_details'].items():
                # Use average confidence score
                avg_confidence = sum(details['confidence_scores']) / len(details['confidence_scores'])
                
                batch_attendance_data.append({
                    'student_name': student_name,
                    'attendance_status': 'present',
                    'confidence_score': avg_confidence
                })
            
            # Record absent students
            for student_name in attendance_data['absent_list']:
                batch_attendance_data.append({
                    'student_name': student_name,
                    'attendance_status': 'absent',
                    'confidence_score': None
                })
            
            # Mark attendance in database
            success_count = self.attendance_manager.mark_batch_attendance(
                batch_attendance_data, 
                session_id=attendance_data['session_id']
            )
            
            logger.info(f"Successfully recorded attendance for {success_count} students")
            
        except Exception as e:
            logger.error(f"Error recording attendance: {e}")
    
    def _display_attendance_summary(self, attendance_data: Dict):
        """
        Display a clear summary of attendance results.
        
        Args:
            attendance_data: Attendance analysis results
        """
        try:
            if not attendance_data:
                logger.error("No attendance data to display")
                return
            
            print("\n" + "=" * 60)
            print("ATTENDANCE SUMMARY")
            print("=" * 60)
            
            # Basic statistics
            print(f"Date: {attendance_data['date']}")
            print(f"Session ID: {attendance_data['session_id']}")
            print(f"Total Students: {attendance_data['total_students']}")
            print(f"Present: {attendance_data['present_students']}")
            print(f"Absent: {attendance_data['absent_students']}")
            print(f"Attendance Rate: {attendance_data['attendance_rate']:.1f}%")
            
            # Present students with confidence scores
            print("\nPRESENT STUDENTS:")
            print("-" * 40)
            if attendance_data['present_details']:
                for student_name, details in attendance_data['present_details'].items():
                    avg_confidence = sum(details['confidence_scores']) / len(details['confidence_scores'])
                    print(f"✓ {student_name:<25} (Confidence: {avg_confidence:.3f})")
            else:
                print("No students marked as present")
            
            # Absent students
            print("\nABSENT STUDENTS:")
            print("-" * 40)
            if attendance_data['absent_list']:
                for student_name in sorted(attendance_data['absent_list']):
                    print(f"✗ {student_name}")
            else:
                print("No students marked as absent")
            
            # Recognition accuracy (if available)
            if hasattr(self.face_recognition, 'calculate_accuracy'):
                try:
                    test_images = self.dataset_loader.get_test_images()
                    if test_images:
                        batch_results = self.face_recognition.batch_recognize(test_images)
                        accuracy_metrics = self.face_recognition.calculate_accuracy(batch_results)
                        
                        print("\nRECOGNITION ACCURACY:")
                        print("-" * 40)
                        print(f"Accuracy: {accuracy_metrics.get('accuracy', 0):.1f}%")
                        print(f"Correct Predictions: {accuracy_metrics.get('correct_predictions', 0)}")
                        print(f"Unknown Predictions: {accuracy_metrics.get('unknown_predictions', 0)}")
                except Exception as e:
                    logger.warning(f"Could not calculate accuracy: {e}")
            
            print("\n" + "=" * 60)
            print("Attendance recording completed successfully!")
            print("=" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"Error displaying attendance summary: {e}")
    
    def get_system_status(self) -> Dict:
        """
        Get current system status and statistics.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Get dataset statistics
            dataset_stats = self.dataset_loader.get_dataset_stats()
            
            # Get database statistics
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_summary = self.attendance_manager.get_attendance_summary(today)
            unique_students = self.attendance_manager.get_unique_students()
            
            status = {
                'dataset_stats': dataset_stats,
                'today_attendance': attendance_summary,
                'total_unique_students': len(unique_students),
                'system_ready': bool(dataset_stats['total_reference_images'] > 0)
            }
            
            return status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}


    def _send_email_notification(self, attendance_data: Dict):
        """
        Send email notification to mentor/advisor.
        
        Args:
            attendance_data: Attendance data with absent student list
        """
        try:
            logger.info("Preparing email notification...")
            
            # Configure email credentials (replace with actual credentials)
            self.email_notifier.configure_credentials(
                sender_email= os.getenv("EMAIL_USER"),
                sender_password=os.getenv("EMAIL_PASSWORD")
            )
            
            # Send notification to mentor/advisor
            recipient_email = "jeyasubhaganesan02@gmail.com"  # Replace with actual email
            
            # Check if absent students exist
            if attendance_data.get('absent_list') and len(attendance_data['absent_list']) > 0:
                logger.info(f"Sending email with {len(attendance_data['absent_list'])} absent students...")
                # Send email with absent students list (existing behavior)
                success = self.email_notifier.send_attendance_notification(
                    recipient_email, 
                    attendance_data
                )
                if success:
                    logger.info("Absent students mail sent successfully")
                else:
                    logger.warning("Failed to send absent students email")
            else:
                logger.info("No absentees found. Email not required.")
                # Optionally send a "all present" notification (commented out as per requirements)
                # success = self.email_notifier.send_attendance_notification(
                #     recipient_email, 
                #     attendance_data
                # )
            
            logger.info("Email notification process completed")
            
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")


def main():
    """
    Main function to run Smart ID attendance system.
    """
    try:
        # Initialize controller
        controller = SmartIDController()
        
        # Display system status
        status = controller.get_system_status()
        logger.info(f"System Status: {status}")
        
        # Run the attendance system
        success = controller.run_attendance_system()
        
        if success:
            logger.info("Smart ID Attendance System completed successfully!")
        else:
            logger.error("Smart ID Attendance System failed!")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return 1


if __name__ == "__main__":
    exit(main())