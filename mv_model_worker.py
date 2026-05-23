"""Model worker for Hunyuan3D-2mv multiview shape generation."""

import base64
import os
import sys
import time
import uuid
from io import BytesIO

import torch
from PIL import Image

sys.path.insert(0, "./hy3dshape")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger


VIEW_KEYS = ("front", "left", "back", "right")


def load_image_from_base64(image: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image)))


class MVModelWorker:
    """Worker for Hunyuan3D-2mv shape-only generation."""

    def __init__(
        self,
        model_path="/work/models/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        device="cuda",
        worker_id=None,
        model_semaphore=None,
        save_dir="gradio_cache/mv",
        mc_algo="mc",
        compile=False,
    ):
        self.model_path = model_path
        self.subfolder = subfolder
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.mc_algo = mc_algo
        self.compile = compile
        self.rembg = None

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"Loading Hunyuan3D-2mv model {model_path}/{subfolder} "
            f"on worker {self.worker_id} ..."
        )
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=False,
            device=device,
        )
        if self.compile:
            self.pipeline.compile()

    def _load_views(self, params):
        views = {}
        if params.get("front"):
            views["front"] = load_image_from_base64(params["front"])
        elif params.get("image"):
            views["front"] = load_image_from_base64(params["image"])

        for key in ("left", "back", "right"):
            if params.get(key):
                views[key] = load_image_from_base64(params[key])

        if not views:
            raise ValueError("No input images provided.")

        if params.get("remove_background", True):
            if self.rembg is None:
                self.rembg = BackgroundRemover()
            for key, image in list(views.items()):
                if image.mode != "RGBA":
                    views[key] = self.rembg(image.convert("RGB")).convert("RGBA")
                else:
                    views[key] = image
        else:
            views = {key: image.convert("RGBA") for key, image in views.items()}

        return {key: views[key] for key in VIEW_KEYS if key in views}

    def get_queue_length(self):
        if self.model_semaphore is None:
            return 0
        return (self.model_semaphore._value if hasattr(self.model_semaphore, "_value") else 0) + (
            len(self.model_semaphore._waiters)
            if hasattr(self.model_semaphore, "_waiters") and self.model_semaphore._waiters is not None
            else 0
        )

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        start_time = time.time()
        logger.info(f"Generating Hunyuan3D-2mv model for uid: {uid}")

        images = self._load_views(params)
        seed = params.get("seed")
        if seed is None:
            seed = uuid.uuid4().int % (2**32)
        seed = int(seed)
        logger.info(f"Using generation seed: {seed}; views: {list(images.keys())}")

        generator = torch.Generator().manual_seed(seed)

        try:
            outputs = self.pipeline(
                image=images,
                num_inference_steps=int(params.get("num_inference_steps", 30)),
                guidance_scale=float(params.get("guidance_scale", 5.0)),
                generator=generator,
                octree_resolution=int(params.get("octree_resolution", 256)),
                num_chunks=int(params.get("num_chunks", 200000)),
                output_type="mesh",
                mc_algo=self.mc_algo,
            )
            mesh = export_to_trimesh(outputs)[0]
            if mesh is None:
                raise ValueError("Model did not produce a valid mesh.")
            logger.info("---MV shape generation takes %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            logger.error(f"MV shape generation failed: {e}")
            raise ValueError(f"Failed to generate 3D mesh: {str(e)}")

        final_save_path = os.path.join(self.save_dir, f"{str(uid)}.glb")
        mesh.export(final_save_path)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("---MV total generation takes %s seconds ---" % (time.time() - start_time))
        return final_save_path, uid
