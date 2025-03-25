import os
import sys

# Add TrajectoryCrafter to Python path if not already there
TRAJ_CRAFTER_PATH = '/root/TrajectoryCrafter'
if TRAJ_CRAFTER_PATH not in sys.path:
    sys.path.append(TRAJ_CRAFTER_PATH)

from inference_ultra import run_trajectory_crafter_with_save, CameraMove

def apply_camera_move(input_path, output_path, camera_move):
    """
    Apply a camera move to a video using TrajectoryCrafter with default parameters.
    
    Args:
        input_path: Path to input video
        output_path: Path to save the output video
        camera_move: Camera movement parameters (use CameraMove constants or custom string)
        
    Returns:
        Path to the final saved video file
    """
    return run_trajectory_crafter_with_save(
        video_path=input_path,
        camera_move=camera_move,
        save_path=output_path,
        stride=2,  # Default
        center_scale=1.0,  # Default
        sampling_steps=5,  # Default
        random_seed=43,  # Default
        mode="gradual"  # Default
    )

# Example usage
if __name__ == "__main__":
    # Input and output paths
    # input_path = "/root/animations_2/pic_1_1.mp4"  # Change this to your input video path
    input_path = "/root/animations_2/pic_3_1.mp4"  # Change this to your input video path
    input_path = '/root/all_reskin_videos/pic_3_3.mp4'
    output_path = "/root/animations_tests/pic_3_6.mp4"  # Change this to your desired output path
    
    # Camera move (use a predefined move from CameraMove class or custom string)
    camera_move = CameraMove.ZOOM_IN  # Example: Use zoom in effect
    camera_move = "0; 180; 0.5; 0; 0"
    # Apply the camera move
    try:
        result = apply_camera_move(input_path, output_path, camera_move)
        print(f"Successfully generated video: {result}")
    except Exception as e:
        print(f"Error processing video: {e}") 

