"""
Dataset Loader for Smart ID - Face Recognition Based Attendance System

This module handles:
1. Downloading LFW dataset automatically
2. Splitting dataset into reference (known students) and test sets using random sampling
3. Organizing images in proper folder structure
4. Supporting repeated testing with different random splits
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.datasets import fetch_lfw_people
from typing import Tuple, List, Dict
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Handles dataset loading and preparation for face recognition system.
    Supports repeated testing through random sampling.
    """
    
    def __init__(self, min_faces_per_person: int = 20, test_split_ratio: float = 0.3):
        """
        Initialize the dataset loader.
        
        Args:
            min_faces_per_person: Minimum images required per person for reliable recognition
            test_split_ratio: Ratio of images to use for testing (0.0 to 1.0)
        """
        self.min_faces_per_person = min_faces_per_person
        self.test_split_ratio = test_split_ratio
        self.base_dir = Path(__file__).parent.parent
        self.dataset_reference_dir = self.base_dir / "dataset_reference"
        self.dataset_test_dir = self.base_dir / "dataset_test"
        
        # Create directories if they don't exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for dataset storage."""
        try:
            self.dataset_reference_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_test_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Dataset directories created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise
    
    def download_and_prepare_lfw_dataset(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Download LFW dataset and split into reference and test sets using random sampling.
        Converts numpy arrays to JPG images and organizes them in proper folder structure.
        
        Returns:
            Tuple of (reference_dict, test_dict) where each dict maps person name to list of image paths
        """
        try:
            logger.info("Downloading LFW dataset...")
            
            # Download LFW dataset with minimum faces filter
            # sklearn provides data as numpy arrays, not folder-based images
            lfw_people = fetch_lfw_people(
                min_faces_per_person=self.min_faces_per_person,
                resize=1.0,  # Keep original size for better recognition
                download_if_missing=True
            )
            
            logger.info(f"Dataset loaded successfully!")
            logger.info(f"Number of people: {len(lfw_people.target_names)}")
            logger.info(f"Total images: {len(lfw_people.images)}")
            
            # Process each person's images with random sampling
            reference_dict = {}
            test_dict = {}
            
            # Group images by person name from numpy arrays
            person_images = {}
            for i, person_idx in enumerate(lfw_people.target):
                person_name = lfw_people.target_names[person_idx]
                
                if person_name not in person_images:
                    person_images[person_name] = []
                
                # Store image array and its index
                person_images[person_name].append({
                    'array': lfw_people.images[i],
                    'index': i
                })
            
            # Process each person with sufficient images
            for person_name, images_data in person_images.items():
                if len(images_data) < 2:
                    logger.warning(f"Insufficient images for {person_name}: {len(images_data)}")
                    continue
                
                # RANDOM SAMPLING: Shuffle images randomly for each run
                random.shuffle(images_data)
                
                # Calculate split point (70% reference, 30% test)
                split_idx = int(len(images_data) * (1 - self.test_split_ratio))
                
                # Split into reference and test sets
                reference_images_data = images_data[:split_idx]
                test_images_data = images_data[split_idx:]
                
                # Convert numpy arrays to JPG and save to directories
                reference_dict[person_name] = self._save_numpy_images_as_jpg(
                    person_name, reference_images_data, self.dataset_reference_dir
                )
                test_dict[person_name] = self._save_numpy_images_as_jpg(
                    person_name, test_images_data, self.dataset_test_dir
                )
                
                logger.info(f"Processed {person_name}: {len(reference_images_data)} reference, {len(test_images_data)} test images")
            
            logger.info("Dataset preparation completed with random sampling!")
            return reference_dict, test_dict
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    def _save_numpy_images_as_jpg(self, person_name: str, images_data: List[Dict], 
                               target_dir: Path) -> List[str]:
        """
        Convert numpy arrays to JPG images and save them to target directory.
        
        Args:
            person_name: Name of the person
            images_data: List of dictionaries containing image arrays and indices
            target_dir: Target directory path
            
        Returns:
            List of saved image paths
        """
        saved_paths = []
        person_target_dir = target_dir / person_name
        person_target_dir.mkdir(exist_ok=True)
        
        for i, image_data in enumerate(images_data):
            try:
                # Convert numpy array to PIL Image
                # sklearn provides images as numpy arrays with values 0-255
                image_array = image_data['array']
                
                # Convert to PIL Image (numpy arrays need to be converted to PIL for saving)
                # This conversion is required because face_recognition.py expects JPG files
                pil_image = Image.fromarray(image_array.astype('uint8'))
                
                # Generate filename
                filename = f"{person_name}_{image_data['index']:04d}.jpg"
                target_path = person_target_dir / filename
                
                # Save as JPG
                pil_image.save(target_path, 'JPEG', quality=95)
                saved_paths.append(str(target_path))
                
            except Exception as e:
                logger.error(f"Failed to save image for {person_name}: {e}")
        
        return saved_paths
    
    def get_reference_images(self) -> Dict[str, List[str]]:
        """Get all reference images from dataset_reference directory."""
        return self._get_images_from_directory(self.dataset_reference_dir)
    
    def get_test_images(self) -> Dict[str, List[str]]:
        """Get all test images from dataset_test directory."""
        return self._get_images_from_directory(self.dataset_test_dir)
    
    def _get_images_from_directory(self, directory: Path) -> Dict[str, List[str]]:
        """
        Helper method to scan directory and return organized image paths.
        
        Args:
            directory: Directory to scan for images
            
        Returns:
            Dictionary mapping person names to list of image paths
        """
        images_dict = {}
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return images_dict
        
        for person_dir in directory.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                image_files = []
                
                for image_file in person_dir.iterdir():
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image_files.append(str(image_file))
                
                if image_files:
                    images_dict[person_name] = sorted(image_files)
        
        return images_dict
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """
        Get statistics about the prepared dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        reference_images = self.get_reference_images()
        test_images = self.get_test_images()
        
        stats = {
            'num_people_reference': len(reference_images),
            'num_people_test': len(test_images),
            'total_reference_images': sum(len(imgs) for imgs in reference_images.values()),
            'total_test_images': sum(len(imgs) for imgs in test_images.values())
        }
        
        return stats
    
    def refresh_test_split(self):
        """
        Refresh the test split with new random sampling.
        This supports repeated testing with different test sets.
        """
        logger.info("Refreshing test split with new random sampling...")
        return self.download_and_prepare_lfw_dataset()


def main():
    """
    Main function to test the dataset loader.
    Demonstrates random sampling for repeated testing.
    """
    try:
        # Initialize dataset loader
        loader = DatasetLoader(min_faces_per_person=20, test_split_ratio=0.3)
        
        # Download and prepare dataset with random sampling
        logger.info("Starting dataset preparation...")
        reference_dict, test_dict = loader.download_and_prepare_lfw_dataset()
        
        # Get statistics
        stats = loader.get_dataset_stats()
        
        logger.info("=== Dataset Statistics ===")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Demonstrate repeated testing capability
        logger.info("\n=== Testing Random Sampling ===")
        logger.info("Running dataset preparation again to demonstrate different random split...")
        reference_dict_2, test_dict_2 = loader.refresh_test_split()
        
        stats_2 = loader.get_dataset_stats()
        logger.info("=== Second Run Statistics ===")
        for key, value in stats_2.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Dataset loader test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
