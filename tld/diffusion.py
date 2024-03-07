import torch
import numpy as np
from tqdm import tqdm
from tld.effnet import EfficientNetEncoder
import tld.danbooru as db

class DiffusionGenerator:
    def __init__(self, model, effnet: EfficientNetEncoder, previewer, device, model_dtype=torch.float32):
        self.model = model
        self.effnet = effnet
        self.previewer = previewer
        self.device = device
        self.model_dtype = model_dtype


    @torch.no_grad()
    def generate(self,
                 n_iter=30,
                 batch=None, #embeddings to condition on
                 num_imgs=4,
                 class_guidance=5,
                 seed=10,  #for reproducibility
                 img_size=16, #height, width of latent
                 exponent=1,
                 seeds=None,
                 noise_levels=None,
                 use_ddpm_plus=True,
                 ):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99

        if use_ddpm_plus:
            lambdas = [np.log((1-sigma)/sigma) for sigma in noise_levels] #log snr
            hs = [lambdas[i] - lambdas[i-1] for i in range(1, len(lambdas))]
            rs = [hs[i-1]/hs[i] for i in range(1, len(hs))]
        labels = torch.cat([db.get_conditions(batch)[:num_imgs], db.get_conditions(batch, True)[:num_imgs],])
        x_t = self.initialize_image(seeds, num_imgs, img_size, seed)
        self.model.eval()

        x0_pred_prev = None

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            
            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if use_ddpm_plus:
                    #x0_pred is a combination of the two previous x0_pred:
                    D = (1+1/(2*rs[i-1]))*x0_pred - (1/(2*rs[i-1]))*x0_pred_prev
                else:
                    #ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance)

        x0_pred_img = self.previewer((x0_pred).to(self.model_dtype))
        return x0_pred_img, x0_pred

    def pred_image(self, noisy_image, labels, noise_level, class_guidance):
        num_imgs = noisy_image.size(0)
        noises = torch.full((2*num_imgs, 1), noise_level)
        x0_pred = self.model(torch.cat([noisy_image, noisy_image]),
                                    noises.to(self.device, self.model_dtype),
                                    labels.to(self.device, self.model_dtype))
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)
        return x0_pred

    def initialize_image(self, seeds, num_imgs, img_size, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            if self.device.type == "xla":
                device = "cpu"
            else:
                device = self.device
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            return torch.randn(num_imgs, 16, img_size, img_size, dtype=self.model_dtype, generator=generator, device=self.device)
        else:
            return seeds.to(self.device, self.model_dtype)

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label