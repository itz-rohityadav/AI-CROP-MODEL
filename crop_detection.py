# crop_detection.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('crop_disease_detector')

class CropDiseaseDetector:
    def __init__(self, model_path=None, class_indices_path=None):
        """Initialize the crop disease detector with a trained model"""
        # Default paths
        if model_path is None:
            model_path = os.path.join('models', 'plant_disease_model_best.keras')
        if class_indices_path is None:
            class_indices_path = os.path.join('models', 'class_indices.json')
        
        # Load the model
        try:
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # Get image size from model input shape
            self.img_size = self.model.input_shape[1]  # Assuming square input
            logger.info(f"Model expects input size: {self.img_size}x{self.img_size}")
            
            # Load class indices
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
                
            # Invert class indices for prediction (handle different formats)
            if all(isinstance(k, str) and k.isdigit() for k in self.class_indices.keys()):
                # Format: {"0": "class_name", "1": "class_name2", ...}
                self.classes = {int(k): v for k, v in self.class_indices.items()}
            elif all(isinstance(v, int) for v in self.class_indices.values()):
                # Format: {"class_name": 0, "class_name2": 1, ...}
                self.classes = {v: k for k, v in self.class_indices.items()}
            else:
                # Assume it's already in the right format
                self.classes = self.class_indices
                
            logger.info(f"Loaded {len(self.classes)} disease classes")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize disease detector: {str(e)}")
    
    def preprocess_image(self, img_path):
        """Preprocess an image for prediction"""
        try:
            # Load and resize image
            img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocess input (same as during training)
            processed_img = preprocess_input(img_array)
            return processed_img
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, img_path):
        """Predict the disease class for an image"""
        try:
            # Check if file exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            # Preprocess the image
            processed_img = self.preprocess_image(img_path)
            
            # Make prediction
            logger.info(f"Making prediction for {img_path}")
            predictions = self.model.predict(processed_img)
            
            # Get the predicted class index and probability
            pred_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_class_idx])
            
            # Get the class name
            if isinstance(pred_class_idx, (np.int64, np.int32)):
                pred_class_idx = int(pred_class_idx)
                
            pred_class = self.classes.get(pred_class_idx, f"Unknown_{pred_class_idx}")
            
            # Create a dictionary of all probabilities
            all_probabilities = {}
            for i, prob in enumerate(predictions[0]):
                class_name = self.classes.get(i, f"Unknown_{i}")
                all_probabilities[class_name] = float(prob)
            
            # Return prediction results
            result = {
                'class': pred_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
            logger.info(f"Prediction result: {pred_class} with confidence {confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return a structured error response that won't break the app
            return {
                'class': 'Error',
                'confidence': 0.0,
                'all_probabilities': {'Error': 1.0},
                'error': str(e)
            }
    
    def get_top_predictions(self, img_path, top_k=3):
        """Get the top k predictions for an image"""
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(img_path)
            
            # Make prediction
            predictions = self.model.predict(processed_img)
            
            # Get top k indices
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            # Create result
            top_predictions = []
            for idx in top_indices:
                class_name = self.classes.get(int(idx), f"Unknown_{idx}")
                confidence = float(predictions[0][idx])
                top_predictions.append({
                    'class': class_name,
                    'confidence': confidence
                })
            
            return top_predictions
        except Exception as e:
            logger.error(f"Error getting top predictions: {str(e)}")
            return [{'class': 'Error', 'confidence': 0.0}]

# For testing the module directly
if __name__ == "__main__":
    try:
        # Initialize detector
        detector = CropDiseaseDetector()
        
        # Test with a sample image if provided
        test_image = "test_image.jpg"  # Replace with an actual test image path
        if os.path.exists(test_image):
            result = detector.predict(test_image)
            print(f"Prediction: {result['class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            # Get top 3 predictions
            top_preds = detector.get_top_predictions(test_image, top_k=3)
            print("\nTop 3 predictions:")
            for i, pred in enumerate(top_preds, 1):
                print(f"{i}. {pred['class']} ({pred['confidence']:.4f})")
        else:
            print(f"Test image not found: {test_image}")
            print("Detector initialized successfully, but no test image was provided.")
    except Exception as e:
        print(f"Error testing detector: {str(e)}")