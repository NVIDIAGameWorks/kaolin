import asyncio
import logging
from typing import Dict, Any, Union, Callable

import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from diffusers import AutoPipelineForInpainting

import kaolin
import kaolin.visualize.web.io
from kaolin.visualize.web.sockets import AsyncMessageHandlerProtocol

from .gaussians import train_splat_subset_colors, gaussians_in_mask

logger = logging.getLogger(__name__)


class GlobalInpaintPipeline:
    pipeline = None

    @staticmethod
    def get():
        if GlobalInpaintPipeline.pipeline is None:
            logger.info(f'Initializing inpainting pipeline')
            GlobalInpaintPipeline.pipeline = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16)
            GlobalInpaintPipeline.pipeline.enable_model_cpu_offload()
        return GlobalInpaintPipeline.pipeline


class TrainGaussiansMessageHandler(AsyncMessageHandlerProtocol):
    def __init__(self, mutable_gsmodel):
        self.gsmodel = mutable_gsmodel
        self.camera = None
        self.device = self.gsmodel.positions.device

    def accepted_message_tags(self) -> list[str]:
        return ['train_gaussians', 'set_camera']

    async def on_message(self, message_tag: str, message_content: Dict,
                         write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        if message_tag == 'set_camera':
            self.camera = kaolin.render.camera.Camera.from_dict(message_content).to(self.device)
        elif message_tag == 'train_gaussians':
            await asyncio.to_thread(self._execute_training_task, message_content, write_message_fn)
        else:
            logger.info(f'Unknown message tag {message_tag}')
            return

    def _execute_training_task(self, message_content, write_message_fn):
        logger.info(f'Optimizing Gaussians')
        # TODO: this should all be one handler, so we can send the rendering back right away

        mask = message_content['mask']
        mask = mask_from_message(mask)
        mask = mask.to(self.device).unsqueeze(0).unsqueeze(-1) > 10  # uint8 --> bool
        gt_image = (message_content['img'].to(self.device).float() / 255).unsqueeze(0)[..., :3]

        gs_mask = gaussians_in_mask(self.camera, self.gsmodel.positions, mask.squeeze())
        new_albedo = train_splat_subset_colors(self.gsmodel, gs_mask,
                                               self.camera, gt_image, mask.repeat(1, 1, 1, 3),
                                               log_every_n=20,
                                               n_steps=300)
        # Write the optimized albedo back into the DC channel of sh_coeff.
        # The ShellRenderer reads gsmodel.sh_coeff fresh on each frame, so this
        # is what the next render-request will pick up.
        with torch.no_grad():
            self.gsmodel.sh_coeff[:, 0:1, :] = new_albedo


class InpaintImageMessageHandler(AsyncMessageHandlerProtocol):
    def __init__(self):
        super().__init__()
        self.image_size = 512
        self.pipeline = GlobalInpaintPipeline.get()

    def accepted_message_tags(self) -> list[str]:
        return ['inpaint']

    async def on_message(self, message_tag: str, message_content: Dict,
                         write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        if message_tag == 'inpaint':
            await asyncio.to_thread(self._execute_inpaint_task, message_content, write_message_fn)
        else:
            logger.info(f'Unknown message tag {message_tag}')
            return

    def _execute_inpaint_task(self, message_content, write_message_fn):
        image = message_content['img']
        image = (image.cuda().float() / 255 * 2 - 1).permute(2, 0, 1).unsqueeze(0)[:, :3, ...]
        image = torchvision.transforms.Resize((self.image_size, self.image_size))(image)
        mask_orig = message_content['mask']
        mask_orig = mask_from_message(mask_orig)
        mask = (mask_orig.cuda().float() / 255).unsqueeze(0).unsqueeze(0)
        mask = torchvision.transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)(mask)
        prompt = message_content['prompt']
        neg_prompt = message_content['neg_prompt']

        image = self.pipeline(prompt=prompt, negative_prompt=neg_prompt, image=image, mask_image=mask).images[0]
        image = torchvision.transforms.Resize((mask_orig.shape[0], mask_orig.shape[1]))(torchvision.transforms.functional.pil_to_tensor(image))
        image = image.clip(0, 255).to(torch.uint8).cpu()

        result = torch.cat([image, mask_orig.unsqueeze(0)], dim=0).permute(1, 2, 0)
        response = {'img': result}
        encoded = kaolin.visualize.web.io.encode_message('inpaint', response, binary=True, image_format='png')
        write_message_fn(encoded, True)



def mask_from_message(mask_orig):
    """ Depending on how it's encoded mask, can be a multi-channel PNG. """
    if len(mask_orig.shape) == 3:  # hwc
        if mask_orig.shape[-1] == 4:
            mask_orig = mask_orig[..., -1]
        else:
            mask_orig = mask_orig.mean(dim=-1)

    return mask_orig