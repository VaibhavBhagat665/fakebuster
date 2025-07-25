import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def create_directories():
    """Create necessary project directories"""
    print("📁 Creating project directories...")
    directories = [
        'data/real_images',
        'data/ai_images',
        'data/text_datasets',
        'saved_models/text_classifier',
        'saved_models/image_classifier',
        'logs',
        'checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Project directories created successfully!")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported!")
        print("Please install Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "datasets>=2.5.0",
        "scikit-learn>=1.1.0",
        "pillow>=9.0.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "flask>=2.2.0",
        "flask-cors>=3.0.10",
        "requests>=2.28.0",
        "tqdm>=4.64.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('>=')[0]}"):
            return False
    
    return True

def create_sample_files():
    """Create sample configuration files"""
    print("📄 Creating sample files...")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Model files
saved_models/
checkpoints/
logs/
*.bin
*.safetensors

# Data
data/
*.csv
*.json

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    
    # Create README
    readme_content = """# AI Detection Models

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
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content.strip())
    
    print("✅ Sample files created!")

def main():
    """Main setup function"""
    print("🚀 AI Detection Models Setup")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    create_directories()
    
    if not install_dependencies():
        print("❌ Failed to install dependencies!")
        print("Please check your internet connection and try again")
        sys.exit(1)
    
    create_sample_files()
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Train your models:")
    print("   python simplified_training.py")
    print("\n2. Start the API server:")
    print("   python flask_api.py")
    print("\n3. Test the API:")
    print("   python test_api.py")
    print("\n4. (Optional) Add your own image datasets:")
    print("   - Real images → data/real_images/")
    print("   - AI images → data/ai_images/")
    
    print("\n💡 Tips:")
    print("- Use a virtual environment: python -m venv env && source env/bin/activate")
    print("- Install GPU support for faster training if you have NVIDIA GPU")
    print("- Check README.md for detailed documentation")
    
    print("\n🔗 Ready for MERN stack integration!")

if __name__ == "__main__":
    main()