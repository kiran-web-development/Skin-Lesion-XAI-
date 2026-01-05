"""
Test script to send image to Flask API and get predictions
"""

import requests
import json
from pathlib import Path

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/predict"

def test_image(image_path):
    """Test image by sending it to Flask API"""
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Error: File not found: {image_path}")
        return
    
    print(f"📸 Testing image: {image_path}")
    print(f"   File size: {image_path.stat().st_size / 1024:.2f} KB")
    print("=" * 60)
    
    try:
        # Open and send image to Flask API
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            print("🔄 Sending request to Flask API...")
            response = requests.post(API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("\n✅ Prediction Successful!")
                print("=" * 60)
                print(f"\n🎯 Predicted Class: {result['predicted_class']}")
                print(f"📊 Confidence: {result['confidence']:.2f}%")
                
                print("\n📋 All Class Probabilities:")
                print("-" * 60)
                for class_name, prob in result['all_probabilities']:
                    bar_length = int(prob / 5)
                    bar = "█" * bar_length
                    print(f"  {class_name:10s} {prob:6.2f}%  {bar}")
                
                print("\n✅ LIME explanation generated successfully!")
                print("   View the visualization in the Flask web interface.")
                
            else:
                print(f"\n❌ Prediction Error: {result.get('error')}")
        else:
            print(f"\n❌ HTTP Error {response.status_code}: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to Flask app at http://127.0.0.1:5000")
        print("   Make sure the Flask app is running!")
    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out (took too long)")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    import sys
    
    # Test image path
    image_path = r"C:\Users\saini\Downloads\original.jpg"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    test_image(image_path)
