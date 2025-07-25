# AI Detection Models

This project provides machine learning models to detect AI-generated content in both text and images.

## Quick Start

1. **Setup**: Run `python setup.py` (you already did this!)
2. **Train Models**: Run `python simplified_training.py`
3. **Start API**: Run `python flask_api.py`
4. **Test API**: Run `python test_api.py`

## Project Structure

- `simplified_training.py` - Train the AI detection models
- `flask_api.py` - REST API server
- `test_api.py` - API testing suite
- `data/` - Training data directory
- `saved_models/` - Trained model storage

## Adding Your Own Data

### Text Data
The training script includes built-in examples, but you can enhance it by:
- Adding more human-written samples to the training data
- Including outputs from different AI models (GPT-4, Claude, Bard, etc.)

### Image Data
1. Add real photos to `data/real_images/`
2. Add AI-generated images to `data/ai_images/`
3. Re-run the training script

## API Endpoints

- `POST /predict-text` - Detect AI-generated text
- `POST /predict-image` - Detect AI-generated images
- `GET /health` - Health check
- `GET /model-info` - Model information

## MERN Stack Integration

The Flask API is designed to work with Express.js backends. See the documentation for integration examples.

## Support

If you encounter issues:
1. Check Python version (3.8+ required)
2. Ensure all dependencies are installed
3. Check that models are trained and saved properly