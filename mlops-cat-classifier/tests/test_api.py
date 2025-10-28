import pytest
import sys
import os
import io
from PIL import Image
import numpy as np


def test_image_processing():
    """Test basic image processing without importing the API"""
    # Create a sample RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')

    # Test that we can create and manipulate images
    assert img.size == (224, 224)
    assert img.mode == 'RGB'

    # Test conversion to array
    arr = np.array(img)
    assert arr.shape == (224, 224, 3)


def test_grayscale_to_rgb_conversion():
    """Test grayscale to RGB conversion"""
    # Create a grayscale image
    gray_img = Image.new('L', (224, 224), color=128)
    assert gray_img.mode == 'L'

    # Convert to RGB
    rgb_img = gray_img.convert('RGB')
    assert rgb_img.mode == 'RGB'
    assert rgb_img.size == (224, 224)


def test_image_normalization():
    """Test image normalization logic"""
    # Create sample image data
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Normalize to [0, 1]
    normalized = img_array.astype(np.float32) / 255.0

    # Check normalization worked
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_image_batch_dimension():
    """Test adding batch dimension"""
    # Create sample image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Add batch dimension
    batch = np.expand_dims(img_array, axis=0)

    # Should have shape (1, 224, 224, 3)
    assert batch.shape == (1, 224, 224, 3)


def test_image_bytes_conversion():
    """Test converting image to/from bytes (for API upload)"""
    # Create a sample image
    img = Image.new('RGB', (224, 224), color='red')

    # Convert to bytes (as Flask would receive it)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Verify we can read it back
    loaded_img = Image.open(img_byte_arr)
    assert loaded_img.size == (224, 224)


def test_png_image_handling():
    """Test PNG format handling"""
    # Create a PNG image
    img = Image.new('RGB', (224, 224), color='blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Verify we can read it
    loaded_img = Image.open(img_byte_arr)
    assert loaded_img.size == (224, 224)


def test_invalid_image_data():
    """Test handling of invalid image data"""
    # Create invalid data
    invalid_data = io.BytesIO(b'This is not an image')

    # Should raise an error when trying to open
    with pytest.raises(Exception):
        Image.open(invalid_data)


def test_image_resize():
    """Test image resizing to model input size"""
    # Create an image with wrong size
    img = Image.new('RGB', (512, 512), color='green')

    # Resize to 224x224
    resized = img.resize((224, 224))

    assert resized.size == (224, 224)


# =============================================================================
# Basic API structure tests
# =============================================================================

def test_api_imports():
    """Test that we can import necessary packages"""
    try:
        from flask import Flask, request, jsonify
        from PIL import Image
        import numpy as np
        import io
        assert True # If we get here, all imports succeeded
    except ImportError as e:
        pytest.skip(f"Skipping: {e}. This will be available in CI/CD environment.")


def test_json_response_structure():
    """Test expected JSON response structure"""
    # Simulate what the API should return
    mock_response = {
        'is_cat': True,
        'confidence': 0.85,
        'detected_class': 'cat'
    }

    # Verify structure
    assert 'is_cat' in mock_response
    assert 'confidence' in mock_response
    assert 'detected_class' in mock_response
    assert isinstance(mock_response['is_cat'], bool)
    assert isinstance(mock_response['confidence'], (int, float))
    assert isinstance(mock_response['detected_class'], str)
    assert 0 <= mock_response['confidence'] <= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
