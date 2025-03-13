import os
import time
import requests
from pathlib import Path

# Base URL for the service
BASE_URL = 'http://127.0.0.1:11234'
print('BASE URL: ', BASE_URL)

# Test video path and output directory
TEST_VIDEO_PATH = '/root/manual.mp4'  # Update this path to your test video
OUT_DIR = Path('test_outputs')


def send_animation_zoom_in_request(video_path: str, output_filename: str) -> None:
    """
    Send a request to the animation_zoom_in endpoint.
    
    Args:
        video_path: Path to the input video file
        output_filename: Where to save the processed video
    """
    url = f'{BASE_URL}/animation_zoom_in/'
    
    # Ensure the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Test video not found at {video_path}")
    
    # Prepare the video file for upload
    files = {
        'video': ('test_video.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    
    try:
        print(f"Sending request to {url} with video='{video_path}'")
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            # Save the processed video
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            print(f'Generated zoom-in animation saved to {output_filename}')
        else:
            print(f'Request failed with status code {response.status_code}: {response.text}')
    
    except Exception as e:
        print(f"Error during request: {e}")
    
    finally:
        # Always close the file
        files['video'][1].close()


def run_test():
    """Run a test of the animation_zoom_in endpoint."""
    # Create output directory if it doesn't exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up test parameters
    test_video = TEST_VIDEO_PATH
    output_video = OUT_DIR / 'test_zoomed_video.mp4'
    
    # Time the request
    start_time = time.time()
    
    try:
        send_animation_zoom_in_request(test_video, str(output_video))
    except Exception as e:
        print(f"Test failed: {e}")
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    run_test() 
