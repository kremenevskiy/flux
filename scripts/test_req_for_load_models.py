import os
import time

import requests

# from dotenv import load_dotenv

# load_dotenv('.env', override=True)

# Replace with the URL where your FastAPI service is running
# BASE_URL = os.getenv('GENERATION_SERVICE_ENDPOINTS_URL')
BASE_URL = 'http://127.0.0.1:11234'
print('BASE URL: ', BASE_URL)

test_image_path = 'assets/nice_frames/frame_1024.png'
out_dir = 'scripts/data_out/'


def send_flux_generate_image_request(prompt, seed, output_filename):
    url = f'{BASE_URL}/flux-generate-image/'
    data = {
        'prompt': prompt,
        'seed': seed,
    }

    print(f"Sending request to {url} with prompt='{prompt}' and seed={seed}")
    response = requests.post(url, data=data)

    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f'Generated image saved to {output_filename}')
    else:
        print(f'Request failed with status code {response.status_code}: {response.text}')


def send_flux_inpaint_image_request(prompt, seed, image_path, mask_path, output_filename):
    url = f'{BASE_URL}/flux-inpaint-image/'
    data = {
        'prompt': prompt,
        'seed': seed,
    }
    files = {
        'image': open(image_path, 'rb'),
        'image_mask': open(mask_path, 'rb'),
    }

    print(
        f"Sending request to {url} with prompt='{prompt}', seed={seed}, image='{image_path}', mask='{mask_path}'"
    )
    response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f'Inpainted image saved to {output_filename}')
    else:
        print(f'Request failed with status code {response.status_code}: {response.text}')

    # Close the file objects
    files['image'].close()
    files['image_mask'].close()


def send_flux_canny_image_request(prompt, seed, image_path, output_filename):
    url = f'{BASE_URL}/flux-canny-image/'
    data = {
        'prompt': prompt,
        'seed': seed,
    }
    files = {
        'image': open(image_path, 'rb'),
    }

    print(f"Sending request to {url} with prompt='{prompt}', seed={seed}, image='{image_path}'")
    response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f'Canny image saved to {output_filename}')
    else:
        print(f'Request failed with status code {response.status_code}: {response.text}')

    # Close the file object
    files['image'].close()


def send_flux_lora_canny_image_request(prompt, seed, image_path, output_filename):
    url = f'{BASE_URL}/flux-canny-lora-image/'
    data = {
        'prompt': prompt,
        'seed': seed,
    }
    files = {
        'image': open(image_path, 'rb'),
    }

    print(f"Sending request to {url} with prompt='{prompt}', seed={seed}, image='{image_path}'")
    response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f'Canny image saved to {output_filename}')
    else:
        print(f'Request failed with status code {response.status_code}: {response.text}')

    # Close the file object
    files['image'].close()


def send_inpaint():
    inpaint_prompt = 'Fill the blank area with a tree'
    inpaint_seed = 123
    input_image_path = test_image_path
    input_mask_path = test_image_path
    inpaint_output = os.path.join(out_dir, 'inpainted_image.png')
    send_flux_inpaint_image_request(
        inpaint_prompt, inpaint_seed, input_image_path, input_mask_path, inpaint_output
    )


def send_generate():
    generate_prompt = 'A sunset over a mountain range'
    generate_seed = 42
    generate_output = os.path.join(out_dir, 'generated_image.png')
    send_flux_generate_image_request(generate_prompt, generate_seed, generate_output)


def send_canny():
    canny_prompt = 'Outline the silhouette of a forest scene'
    canny_seed = 321
    input_image_path = test_image_path
    canny_output = os.path.join(out_dir, 'canny_image.png')
    send_flux_canny_image_request(canny_prompt, canny_seed, input_image_path, canny_output)


def send_lora_canny():
    canny_prompt = 'Outline the silhouette of a forest scene'
    canny_seed = 321
    input_image_path = test_image_path
    canny_output = os.path.join(out_dir, 'canny_lora_image.png')
    send_flux_lora_canny_image_request(canny_prompt, canny_seed, input_image_path, canny_output)


if __name__ == '__main__':
    start = time.time()
    # Example usage for flux-generate-image endpoint

    # send_lora_canny()

    send_canny()
    send_inpaint()
    send_generate()
    send_lora_canny()

    # send_generate()
    # send_inpaint()
    # send_inpaint()

    # send_canny()
    # send_canny()
    # send_canny()

    # send_generate()
    # send_inpaint()
    # send_canny()

    # send_inpaint()
    # send_canny()
    # send_generate()

    # send_canny()
    # send_canny()
    # send_canny()
    # send_canny()

    # send_inpaint()
    # send_inpaint()
    # send_inpaint()
    # send_inpaint()
    # send_inpaint()
    # send_inpaint()

    # send_generate()
    # send_generate()
    # send_generate()
    # send_generate()
    # send_generate()

    # send_inpaint()

    print('total: ', time.time() - start)
