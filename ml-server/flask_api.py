# Flask API for AI Detection Models
# Requirements: pip install flask flask-cors torch transformers pillow numpy

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import io
import base64
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for MERN stack integration

# ======================= MODEL CLASSES =======================

class TextClassifier(nn.Module):
    """Simplified RoBERTa-based text classifier for AI detection"""
    def __init__(self, model_name="roberta-base", num_classes=2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # First token [CLS]
        output = self.dropout(cls_output)
        return self.classifier(output)

# ======================= GLOBAL VARIABLES =======================

text_model = None
text_tokenizer = None
image_model = None
image_processor = None

# ======================= MODEL LOADING =======================

def load_models():
    """Load both models on startup"""
    global text_model, text_tokenizer, image_model, image_processor
    
    try:
        # Load text classifier
        logger.info("Loading text classifier...")
        text_model_path = './saved_models/text_classifier'
        
        if os.path.exists(text_model_path):
            text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            text_model = TextClassifier()
            
            # Load the trained weights if available
            try:
                text_model.load_state_dict(torch.load(
                    os.path.join(text_model_path, 'pytorch_model.bin'),
                    map_location=torch.device('cpu')
                ))
                text_model.eval()
                logger.info("Text classifier loaded successfully!")
            except:
                logger.warning("Trained weights not found, using pre-trained RoBERTa")
                text_model.eval()
        else:
            logger.warning("Text classifier not found, loading base RoBERTa")
            text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            text_model = TextClassifier()
            text_model.eval()
        
        # Load image classifier
        logger.info("Loading image classifier...")
        image_model_path = './saved_models/image_classifier'
        
        if os.path.exists(image_model_path):
            image_processor = AutoImageProcessor.from_pretrained(image_model_path)
            image_model = AutoModelForImageClassification.from_pretrained(
                image_model_path,
                num_labels=2
            )
            image_model.eval()
            logger.info("Image classifier loaded successfully!")
        else:
            logger.warning("Image classifier not found, loading base ViT")
            image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            image_model = AutoModelForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            image_model.eval()
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

# ======================= PREDICTION FUNCTIONS =======================

def predict_text(text):
    """Predict if text is AI-generated or human-written"""
    try:
        # Tokenize input
        inputs = text_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,  # Match training max_length
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = text_model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        # Convert prediction to label
        label = "AI-generated" if prediction == 1 else "Human-written"
        
        return {
            'prediction': label,
            'confidence': float(confidence),
            'probabilities': {
                'human': float(probabilities[0][0]),
                'ai': float(probabilities[0][1])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in text prediction: {str(e)}")
        raise e

def predict_image(image):
    """Predict if image is AI-generated or real"""
    try:
        # Process image
        inputs = image_processor(image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = image_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        # Convert prediction to label
        label = "AI-generated" if prediction == 1 else "Real"
        
        return {
            'prediction': label,
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probabilities[0][0]),
                'ai': float(probabilities[0][1])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        raise e

# ======================= API ROUTES (FIXED ENDPOINTS) =======================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'text': text_model is not None,
            'image': image_model is not None
        }
    })

# Fixed endpoint to match frontend expectations
@app.route('/api/detect/text', methods=['POST'])
def detect_text():
    """Text detection endpoint - matches frontend calls"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text must be a non-empty string'
            }), 400
        
        if len(text) > 10000:  # Limit text length
            return jsonify({
                'error': 'Text too long (max 10,000 characters)'
            }), 400
        
        # Make prediction
        result = predict_text(text)
        
        return jsonify({
            'success': True,
            'isAI': result['prediction'] == 'AI-generated',
            'confidence': result['confidence'],
            'details': result,
            'input_length': len(text)
        })
        
    except Exception as e:
        logger.error(f"Error in text detection endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Fixed endpoint to match frontend expectations
@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Image detection endpoint - matches frontend calls"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            # Check for base64 encoded image in JSON
            data = request.get_json()
            if data and 'image' in data:
                try:
                    # Handle base64 image
                    image_data = data['image']
                    if image_data.startswith('data:image/'):
                        # Remove data URL prefix
                        image_data = image_data.split(',')[1]
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid base64 image: {str(e)}'
                    }), 400
            else:
                return jsonify({
                    'success': False,
                    'error': 'No image provided'
                }), 400
        else:
            # Handle file upload
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Check file type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            file_extension = file.filename.lower().split('.')[-1]
            
            if file_extension not in allowed_extensions:
                return jsonify({
                    'success': False,
                    'error': f'Unsupported file type: {file_extension}'
                }), 400
            
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid image file: {str(e)}'
                }), 400
        
        # Check image size (limit to reasonable dimensions)
        if image.size[0] > 4096 or image.size[1] > 4096:
            return jsonify({
                'success': False,
                'error': 'Image too large (max 4096x4096 pixels)'
            }), 400
        
        # Make prediction
        result = predict_image(image)
        
        return jsonify({
            'success': True,
            'isAI': result['prediction'] == 'AI-generated',
            'confidence': result['confidence'],
            'details': result,
            'image_size': image.size
        })
        
    except Exception as e:
        logger.error(f"Error in image detection endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Keep original endpoints for backward compatibility
@app.route('/predict-text', methods=['POST'])
def predict_text_endpoint():
    """Original text prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text must be a non-empty string'
            }), 400
        
        if len(text) > 10000:
            return jsonify({
                'error': 'Text too long (max 10,000 characters)'
            }), 400
        
        result = predict_text(text)
        
        return jsonify({
            'success': True,
            'result': result,
            'input_length': len(text)
        })
        
    except Exception as e:
        logger.error(f"Error in text prediction endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict-image', methods=['POST'])
def predict_image_endpoint():
    """Original image prediction endpoint"""
    try:
        if 'image' not in request.files:
            data = request.get_json()
            if data and 'image' in data:
                try:
                    image_data = base64.b64decode(data['image'])
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                except Exception as e:
                    return jsonify({
                        'error': f'Invalid base64 image: {str(e)}'
                    }), 400
            else:
                return jsonify({
                    'error': 'No image provided'
                }), 400
        else:
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected'
                }), 400
            
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            file_extension = file.filename.lower().split('.')[-1]
            
            if file_extension not in allowed_extensions:
                return jsonify({
                    'error': f'Unsupported file type: {file_extension}'
                }), 400
            
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({
                    'error': f'Invalid image file: {str(e)}'
                }), 400
        
        if image.size[0] > 4096 or image.size[1] > 4096:
            return jsonify({
                'error': 'Image too large (max 4096x4096 pixels)'
            }), 400
        
        result = predict_image(image)
        
        return jsonify({
            'success': True,
            'result': result,
            'image_size': image.size
        })
        
    except Exception as e:
        logger.error(f"Error in image prediction endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch-predict-text', methods=['POST'])
def batch_predict_text():
    """Batch text prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'error': 'texts must be a non-empty list'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'error': 'Batch size too large (max 100 texts)'
            }), 400
        
        results = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) == 0:
                results.append({
                    'index': i,
                    'error': 'Invalid text'
                })
                continue
            
            try:
                result = predict_text(text)
                results.append({
                    'index': i,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch text prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    return jsonify({
        'text_model': {
            'loaded': text_model is not None,
            'architecture': 'RoBERTa-based classifier',
            'max_length': 256
        },
        'image_model': {
            'loaded': image_model is not None,
            'architecture': 'Vision Transformer (ViT)',
            'max_size': '4096x4096'
        },
        'supported_formats': {
            'text': ['string'],
            'image': ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']
        }
    })

# ======================= ERROR HANDLERS =======================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'success': False,
        'error': 'Payload too large'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ======================= MAIN =======================

if __name__ == '__main__':
    logger.info("Starting Flask API server...")
    load_models()
    logger.info("Models loaded successfully!")
    
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
    
    app.run(
        host='0.0.0.0',
        port=7000,  
        debug=True  
    )