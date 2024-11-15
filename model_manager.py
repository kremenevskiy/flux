# model_manager.py


import torch


class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model = None

    def get_model(self, model_name: str):
        if self.current_model_name == model_name and self.model is not None:
            return self.model

        # Unload the current model if it exists
        if self.model is not None:
            self.unload_model()
        # Load the requested model
        if model_name == 'flux_generate':
            self.model = self.load_flux_generate_model()
        elif model_name == 'flux_inpaint':
            self.model = self.load_flux_inpaint_model()
        else:
            raise ValueError(f'Unknown model name: {model_name}')
        self.current_model_name = model_name
        return self.model

    def unload_model(self) -> None:
        # Unload the current model and clear CUDA cache
        del self.model
        self.model = None
        self.current_model_name = None
        torch.cuda.empty_cache()

    def load_flux_generate_model(self):
        # Load the flux_generate model
        from flux_base.flux_generate import get_pipeline

        pipe, good_vae = get_pipeline()
        return (pipe, good_vae)

    def load_flux_inpaint_model(self):
        # Load the flux_inpaint model
        from flux_inpaint.flux_inpaint import get_model_pipe

        pipe = get_model_pipe()
        return pipe
