import cv2
import os
from main import FaceSystem

def add_sample_faces():
    """Add 3 sample faces to the database."""
    print("Initializing Face System...")
    face_system = FaceSystem()
    
    # List of people to register
    people = [
        {"name": "Basel Anaya", "image_path": "faces/Basel.jpg"},
        {"name": "Ahmad Al Zoubi", "image_path": "faces/Ahmad.jpg"},
        {"name": "Mohammad Jammous", "image_path": "faces/Mohammad.png"}
    ]
    
    # Create faces directory if it doesn't exist
    os.makedirs("faces", exist_ok=True)
    
    print("\nPlease place the ID card images in the 'faces' directory with names:")
    for person in people:
        print(f"- {person['image_path']}")
    
    input("\nPress Enter when you have added the images...")
    
    # Register each person
    for person in people:
        print(f"\nProcessing {person['name']}...")
        
        # Check if image exists
        if not os.path.exists(person['image_path']):
            print(f"Error: Image {person['image_path']} not found")
            continue
        
        # Read image
        image = cv2.imread(person['image_path'])
        if image is None:
            print(f"Error: Could not read image {person['image_path']}")
            continue
        
        # Register face
        if face_system.register_face_from_id_card(image, person['name']):
            print(f"Successfully registered {person['name']}")
        else:
            print(f"Failed to register {person['name']}")

if __name__ == "__main__":
    print("Face Registration Script")
    print("=======================")
    add_sample_faces()
    print("\nDone!") 