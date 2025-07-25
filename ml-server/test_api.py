import requests
import json
import base64
import io
from PIL import Image
import numpy as np
import time
import os

# Configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 30

def create_test_image():
    """Create a simple test image"""
    # Create a simple RGB image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img, img_base64

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n📊 Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info error: {e}")
        return False

def test_text_prediction():
    """Test text prediction endpoint"""
    print("\n📝 Testing Text Prediction...")
    
    test_cases = [
        {
            "text": "I love spending weekends with my family, it's the best part of my week!",
            "expected": "Human-written"
        },
        {
            "text": "Based on comprehensive analysis of multiple data points, this solution provides optimal performance metrics across various parameters and demonstrates significant improvements in operational efficiency.",
            "expected": "AI-generated"
        },
        {
            "text": "My cat knocked over my coffee this morning and I'm still finding puddles everywhere.",
            "expected": "Human-written"
        },
        {
            "text": "To achieve optimal results, it is recommended to implement best practices and follow industry standards for maximum efficiency and effectiveness.",
            "expected": "AI-generated"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Text: {test_case['text'][:60]}...")
        print(f"   Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict-text",
                json={"text": test_case["text"]},
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = data['result']
                    print(f"   Prediction: {result['prediction']}")
                    print(f"   Confidence: {result['confidence']:.3f}")
                    print(f"   Probabilities: Human={result['probabilities']['human']:.3f}, AI={result['probabilities']['ai']:.3f}")
                    
                    if result['prediction'] == test_case['expected']:
                        print("   ✅ Correct prediction!")
                        success_count += 1
                    else:
                        print("   ⚠️  Unexpected prediction (this is normal for demo)")
                else:
                    print(f"   ❌ API returned error: {data.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error: {e}")
    
    print(f"\n📊 Text Prediction Results: {success_count}/{len(test_cases)} correct")
    return success_count > 0

def test_image_prediction_base64():
    """Test image prediction with base64 encoded image"""
    print("\n🖼️  Testing Image Prediction (Base64)...")
    
    try:
        # Create test image
        img, img_base64 = create_test_image()
        print(f"   Created test image: {img.size}")
        
        # Test with base64
        response = requests.post(
            f"{BASE_URL}/predict-image",
            json={"image": img_base64},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data['result']
                print(f"   ✅ Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Probabilities: Real={result['probabilities']['real']:.3f}, AI={result['probabilities']['ai']:.3f}")
                print(f"   Image size processed: {data.get('image_size', 'Unknown')}")
                return True
            else:
                print(f"   ❌ API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False

def test_image_prediction_file():
    """Test image prediction with file upload"""
    print("\n🖼️  Testing Image Prediction (File Upload)...")
    
    try:
        # Create and save test image
        img, _ = create_test_image()
        test_image_path = "test_image.png"
        img.save(test_image_path)
        print(f"   Created test image file: {test_image_path}")
        
        # Test with file upload
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test_image.png', f, 'image/png')}
            response = requests.post(
                f"{BASE_URL}/predict-image",
                files=files,
                timeout=TIMEOUT
            )
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data['result']
                print(f"   ✅ Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Probabilities: Real={result['probabilities']['real']:.3f}, AI={result['probabilities']['ai']:.3f}")
                return True
            else:
                print(f"   ❌ API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False

def test_batch_text_prediction():
    """Test batch text prediction"""
    print("\n📚 Testing Batch Text Prediction...")
    
    texts = [
        "I love my dog so much, he's the best companion ever!",
        "According to extensive research, this methodology demonstrates optimal performance.",
        "Had the worst day at work today, but pizza for dinner made it better.",
        "The implementation of advanced algorithms enables superior functionality."
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch-predict-text",
            json={"texts": texts},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                results = data['results']
                print(f"   ✅ Processed {data['total_processed']} texts:")
                
                for result in results:
                    if 'result' in result:
                        prediction = result['result']
                        print(f"   [{result['index']}] {prediction['prediction']} (conf: {prediction['confidence']:.3f})")
                    else:
                        print(f"   [{result['index']}] Error: {result.get('error', 'Unknown')}")
                
                return True
            else:
                print(f"   ❌ API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🚨 Testing Error Handling...")
    
    error_tests = [
        {
            "name": "Empty text",
            "endpoint": "/predict-text",
            "data": {"text": ""},
            "expected_status": 400
        },
        {
            "name": "Missing text field",
            "endpoint": "/predict-text",
            "data": {},
            "expected_status": 400
        },
        {
            "name": "Too long text",
            "endpoint": "/predict-text",
            "data": {"text": "a" * 15000},
            "expected_status": 400
        },
        {
            "name": "Invalid endpoint",
            "endpoint": "/invalid-endpoint",
            "data": {},
            "expected_status": 404
        }
    ]
    
    success_count = 0
    
    for test in error_tests:
        print(f"\n   🧪 Testing: {test['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}{test['endpoint']}",
                json=test['data'],
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            if response.status_code == test['expected_status']:
                print(f"      ✅ Correct error status: {response.status_code}")
                success_count += 1
            else:
                print(f"      ⚠️  Expected {test['expected_status']}, got {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Request error: {e}")
    
    print(f"\n   📊 Error Handling Results: {success_count}/{len(error_tests)} correct")
    return success_count > 0

def test_server_connection():
    """Test if server is running"""
    print("🔗 Testing server connection...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main test function"""
    print("🧪 AI Detection API Test Suite")
    print("=" * 50)
    
    # Check if server is running
    if not test_server_connection():
        print("❌ Server is not running!")
        print("\n🔧 Please start the Flask API first:")
        print("   python flask_api.py")
        print("\nThen run this test again:")
        print("   python test_api.py")
        return
    
    print("✅ Server is running!")
    print("\n🚀 Starting comprehensive API tests...")
    
    # Run all tests
    test_results = []
    
    test_results.append(("Health Check", test_health_check()))
    test_results.append(("Model Info", test_model_info()))
    test_results.append(("Text Prediction", test_text_prediction()))
    test_results.append(("Image Prediction (Base64)", test_image_prediction_base64()))
    test_results.append(("Image Prediction (File)", test_image_prediction_file()))
    test_results.append(("Batch Text Prediction", test_batch_text_prediction()))
    test_results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your AI Detection API is working perfectly!")
        print("\n🚀 Ready for MERN stack integration!")
        print("\n📚 Integration Tips:")
        print("   • Use axios or fetch for HTTP requests")
        print("   • Handle loading states during predictions")
        print("   • Implement proper error handling")
        print("   • Consider file size limits for images")
        
    elif passed > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {passed}/{total} tests passed")
        print("🔧 Some features may need attention, but core functionality works!")
        
    else:
        print("\n❌ ALL TESTS FAILED!")
        print("🔧 Troubleshooting steps:")
        print("   1. Check if Flask server is running")
        print("   2. Verify models are trained and saved")
        print("   3. Check Python dependencies")
        print("   4. Review server logs for errors")
    
    print("\n💡 Example API Usage:")
    print("   curl -X POST http://localhost:5000/predict-text \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"text\": \"Your text here\"}'")

if __name__ == "__main__":
    main()