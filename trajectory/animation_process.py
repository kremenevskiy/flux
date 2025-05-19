import os
import sys
import subprocess
from pathlib import Path
import shutil

# Add TrajectoryCrafter to Python path if not already there
TRAJ_CRAFTER_PATH = '/root/TrajectoryCrafter'
if TRAJ_CRAFTER_PATH not in sys.path:
    sys.path.append(TRAJ_CRAFTER_PATH)
def process_zoom_in_animation(
    input_video_path: Path,
    output_video_path: Path,
    zoom_params: str, 
    sampling_steps: int, 
    venv_python: str = '/root/TrajectoryCrafter/venv/bin/python'
) -> Path:
    """
    Process a video with zoom-in animation effect using TrajectoryCrafter.
    
    Args:
        input_video_path: Path to the input video file
        output_video_path: Path where to save the processed video
        venv_python: Path to the Python interpreter in TrajectoryCrafter's venv
        
    Returns:
        Path to the processed video file
    """
    try:
        # Get the script path relative to this file
        script_path = Path(__file__).parent / 'inference_ultra.py'
        
        # Ensure output directory exists
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up environment with TrajectoryCrafter in Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{TRAJ_CRAFTER_PATH}:{env.get('PYTHONPATH', '')}"
        
        # Run the inference script
        subprocess.run([
            venv_python,
            str(script_path),
            '--video_path', str(input_video_path),
            '--camera_move', str(zoom_params),  # ZOOM_IN parameters
            '--sampling_steps', str(sampling_steps),
            '--mode', 'gradual',
            '--save_path', str(output_video_path)
        ], check=True, env=env)
        
        return output_video_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error running TrajectoryCrafter: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during video processing: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    input_path = Path("/root/flux/data/animation/dfe44/base_video.mp4")
    output_path = Path("zoomed_test_video_2.mp4")
    
    result_path = process_zoom_in_animation(input_path, output_path)
    print(f"Video processed successfully: {result_path}")

