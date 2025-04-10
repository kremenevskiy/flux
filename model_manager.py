# model_manager.py


import torch
import gc



class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model = None

    def get_model(self, model_name: str, **kwargs):
        if self.current_model_name == model_name and self.model is not None:
            return self.model

        # Unload the current model if it exists
        if self.model is not None:
            self.unload_model()
        # Load the requested model
        if model_name == 'flux_generate':
            self.model = self.load_flux_generate_model()
        elif model_name == 'flux_generate_with_lora':
            self.model = self.load_flux_with_lora()
        elif model_name == 'flux_inpaint':
            self.model = self.load_flux_inpaint_model()
        elif model_name == 'flux_canny':
            self.model = self.load_flux_canny_model()
        elif model_name == 'flux_lora':
            self.model = self.load_flux_lora_model()
        elif model_name == 'flux_lora_canny':
            self.model = self.load_flux_lora_canny_model()
        else:
            raise ValueError(f'Unknown model name: {model_name}')
        self.current_model_name = model_name
        return self.model

    def unload_model(self) -> None:
        # Unload the current model and clear CUDA cache
        if isinstance(self.model, tuple):
            for sub_model in self.model:
                del sub_model  
        del self.model
        self.model = None
        self.current_model_name = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def load_flux_generate_model(self):
        # Load the flux_generate model
        from flux_base.flux_generate import get_model_pipe

        pipe, good_vae = get_model_pipe()
        return (pipe, good_vae)

    def load_flux_inpaint_model(self):
        # Load the flux_inpaint model
        from flux_inpaint.flux_inpaint import get_model_pipe

        pipe = get_model_pipe()
        return pipe

    def load_flux_canny_model(self):
        # Load the flux_inpaint model
        from flux_canny.flux_canny import get_model_pipe

        pipe = get_model_pipe()
        return pipe

    def load_flux_lora_model(self):
        # Load the flux_inpaint model
        from flux_lora.inference_lora import get_model_pipe

        pipe = get_model_pipe()
        return pipe

    def load_flux_lora_canny_model(self):
        # Load the flux_inpaint model
        from flux_lora.comfy_inf_2 import get_model_pipe

        pipe = get_model_pipe()
        return pipe


    def load_flux_with_lora(self):
        from flux_base.flux_generate import get_model_pipe_with_lora

        pipe = get_model_pipe_with_lora()
        return pipe
