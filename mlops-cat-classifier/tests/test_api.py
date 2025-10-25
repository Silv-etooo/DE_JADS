"""
Unit tests for Cat Classifier Prediction API
"""

import pytest
import sys
import os
import io
from PIL import Image
import numpy as np

# Add parent directory to path to import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after path is set
from app.api.predict_api import app, prepare_image, is_cat


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing"""
    # Create a 224x224 RGB image with random pixels
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes"""
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


# =============================================================================
# Test Helper Functions
# =============================================================================

def test_prepare_image_shape(sample_image):
    """Test that prepare_image returns correct shape"""
    processed = prepare_image(sample_image)

    # Should return (1, 224, 224, 3) - batch of 1 image
    assert processed.shape == (1, 224, 224, 3), \
        f"Expected shape (1, 224, 224, 3), got {processed.shape}"


def test_prepare_image_normalization(sample_image):
    """Test that prepare_image normalizes pixel values correctly"""
    processed = prepare_image(sample_image)

    # MobileNetV2 preprocessing scales to [-1, 1]
    assert processed.min() >= -1.0, "Pixel values should be >= -1"
    assert processed.max() <= 1.0, "Pixel values should be <= 1"


def test_prepare_image_rgb_conversion():
    """Test that grayscale images are converted to RGB"""
    # Create a grayscale image
    gray_img = Image.new('L', (224, 224), color=128)

    # This should work without errors (converts to RGB internally)
    processed = prepare_image(gray_img)
    assert processed.shape == (1, 224, 224, 3)


def test_is_cat_function():
    """Test the is_cat detection logic"""
    # Mock predictions with a cat class
    mock_predictions = [[
        ('n02123045', 'tabby', 0.85),
        ('n02123159', 'tiger_cat', 0.10),
        ('n02124075', 'Egyptian_cat', 0.03),
    ]]

    is_cat_detected, confidence, label = is_cat(mock_predictions)

    assert is_cat_detected == True, "Should detect cat"
    assert confidence == 0.85, f"Expected confidence 0.85, got {confidence}"
    assert label == 'tabby', f"Expected label 'tabby', got {label}"


def test_is_cat_function_no_cat():
    """Test is_cat when no cat is detected"""
    # Mock predictions with no cat classes
    mock_predictions = [[
        ('n02084071', 'dog', 0.90),
        ('n02121808', 'domestic_dog', 0.05),
        ('n02110063', 'poodle', 0.03),
    ]]

    is_cat_detected, confidence, label = is_cat(mock_predictions)

    assert is_cat_detected == False, "Should not detect cat"
    assert confidence == 0.0, f"Expected confidence 0.0, got {confidence}"
    assert label == "not a cat", f"Expected 'not a cat', got {label}"


# =============================================================================
# Test API Endpoints
# =============================================================================

def test_health_endpoint(client):
    """Test the /health endpoint"""
    response = client.get('/health')

    assert response.status_code == 200, "Health endpoint should return 200"

    data = response.get_json()
    assert 'status' in data, "Response should contain 'status'"
    assert data['status'] == 'healthy', "Status should be 'healthy'"
    assert 'model' in data, "Response should contain 'model'"
    assert data['model'] == 'MobileNetV2', "Model should be MobileNetV2"


def test_home_endpoint(client):
    """Test the / endpoint returns HTML"""
    response = client.get('/')

    assert response.status_code == 200, "Home endpoint should return 200"
    assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data, \
        "Should return HTML content"


def test_predict_endpoint_no_file(client):
    """Test /predict endpoint with no file"""
    response = client.post('/predict')

    assert response.status_code == 400, "Should return 400 for no file"

    data = response.get_json()
    assert 'error' in data, "Response should contain 'error'"
    assert data['error'] == 'No File Provided', "Should indicate no file provided"


def test_predict_endpoint_empty_filename(client):
    """Test /predict endpoint with empty filename"""
    data = {
        'file': (io.BytesIO(b''), '')  # Empty filename
    }

    response = client.post('/predict', data=data, content_type='multipart/form-data')

    assert response.status_code == 400, "Should return 400 for empty filename"

    json_data = response.get_json()
    assert 'error' in json_data, "Response should contain 'error'"


def test_predict_endpoint_valid_image(client, sample_image_bytes):
    """Test /predict endpoint with valid image"""
    data = {
        'file': (sample_image_bytes, 'test_cat.jpg')
    }

    response = client.post('/predict', data=data, content_type='multipart/form-data')

    assert response.status_code == 200, f"Should return 200, got {response.status_code}"

    json_data = response.get_json()

    # Check response structure
    assert 'is_cat' in json_data, "Response should contain 'is_cat'"
    assert 'confidence' in json_data, "Response should contain 'confidence'"
    assert 'detected_class' in json_data, "Response should contain 'detected_class'"
    assert 'top_predictions' in json_data, "Response should contain 'top_predictions'"

    # Check data types
    assert isinstance(json_data['is_cat'], bool), "'is_cat' should be boolean"
    assert isinstance(json_data['confidence'], (int, float)), "'confidence' should be numeric"
    assert isinstance(json_data['detected_class'], str), "'detected_class' should be string"
    assert isinstance(json_data['top_predictions'], list), "'top_predictions' should be list"

    # Check predictions list structure
    if len(json_data['top_predictions']) > 0:
        pred = json_data['top_predictions'][0]
        assert 'class' in pred, "Prediction should contain 'class'"
        assert 'confidence' in pred, "Prediction should contain 'confidence'"


def test_predict_endpoint_png_image(client):
    """Test /predict endpoint with PNG image"""
    # Create a PNG image
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    data = {
        'file': (img_byte_arr, 'test_image.png')
    }

    response = client.post('/predict', data=data, content_type='multipart/form-data')

    assert response.status_code == 200, "Should accept PNG images"


def test_predict_endpoint_grayscale_image(client):
    """Test /predict endpoint with grayscale image"""
    # Create a grayscale image
    img = Image.new('L', (224, 224), color=128)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {
        'file': (img_byte_arr, 'gray_image.jpg')
    }

    response = client.post('/predict', data=data, content_type='multipart/form-data')

    # Should convert grayscale to RGB and process successfully
    assert response.status_code == 200, "Should handle grayscale images"


def test_predict_endpoint_invalid_file(client):
    """Test /predict endpoint with invalid file (not an image)"""
    # Create invalid file content
    invalid_data = io.BytesIO(b'This is not an image file')

    data = {
        'file': (invalid_data, 'not_an_image.txt')
    }

    response = client.post('/predict', data=data, content_type='multipart/form-data')

    # Should return 500 error for invalid image
    assert response.status_code == 500, "Should return 500 for invalid image"

    json_data = response.get_json()
    assert 'error' in json_data, "Response should contain 'error'"
    assert json_data['error'] == 'Prediction failed', "Should indicate prediction failure"


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_prediction_workflow(client, sample_image_bytes):
    """Test complete prediction workflow"""
    # 1. Check health
    health_response = client.get('/health')
    assert health_response.status_code == 200

    # 2. Make prediction
    data = {
        'file': (sample_image_bytes, 'cat_test.jpg')
    }

    predict_response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert predict_response.status_code == 200

    # 3. Verify prediction response
    result = predict_response.get_json()
    assert 'is_cat' in result
    assert 'top_predictions' in result

    # Confidence should be between 0 and 1
    if result['confidence'] > 0:
        assert 0 <= result['confidence'] <= 1, "Confidence should be between 0 and 1"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
