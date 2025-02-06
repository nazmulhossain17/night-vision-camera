import cv2
import numpy as np
from ultralytics import YOLO

def military_night_vision(frame):
    """
    Apply military-style night vision effect with phosphor green color
    and enhanced resolution
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for contrast enhancement
    enhanced = cv2.equalizeHist(gray)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance dark regions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    
    # Create the phosphor green effect
    b_channel = np.zeros_like(enhanced)
    g_channel = enhanced
    r_channel = np.zeros_like(enhanced)
    
    # Combine channels
    result = cv2.merge([b_channel, g_channel, r_channel])
    
    # Apply slight blur for authentic look
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    # Optional: sharpen the image to enhance clarity
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)
    
    return result

def main():
    # Initialize camera with higher resolution
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Set higher resolution (1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Set higher ISO (if supported by your camera, or adjust exposure)
    # Use the camera's exposure settings for dark environments
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Set exposure time to a higher value if supported
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            
            if not success:
                print("Failed to read frame")
                break
            
            # Apply military night vision effect
            enhanced_frame = military_night_vision(frame)
            
            # Display both original and enhanced frames
            cv2.imshow('Original', frame)
            cv2.imshow('Military Night Vision', enhanced_frame)
            
            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
