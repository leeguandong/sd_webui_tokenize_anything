import time
import torch
import numpy as np

from tokenize_anything import engine
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from tokenize_anything.models.easy_build import model_registry


class Predictor(object):
    """Predictor."""

    def __init__(self, model, concept_weights):
        self.model = model
        self.batch_size = 256
        self.model.concept_projector.reset_weights(concept_weights)
        self.model.text_decoder.reset_cache(max_batch_size=self.batch_size)

    def preprocess_images(self, imgs):
        """Preprocess the inference images."""
        im_batch, im_shapes, im_scales = [], [], []
        for img in imgs:
            scaled_imgs, scales = im_rescale(img, scales=[1024])
            im_batch += scaled_imgs
            im_scales += scales
            im_shapes += [x.shape[:2] for x in scaled_imgs]
        im_batch = im_vstack(im_batch, self.model.pixel_mean_value, size=(1024, 1024))
        im_shapes = np.array(im_shapes)
        im_scales = np.array(im_scales).reshape((len(im_batch), -1))
        im_info = np.hstack([im_shapes, im_scales]).astype("float32")
        return im_batch, im_info

    @torch.inference_mode()
    def get_results(self, examples):
        """Return the results."""
        # Preprocess images and prompts.
        imgs = [example["img"] for example in examples]
        points = np.concatenate([example["points"] for example in examples])
        im_batch, im_info = self.preprocess_images(imgs)
        num_prompts = points.shape[0] if len(points.shape) > 2 else 1
        batch_shape = im_batch.shape[0], num_prompts // im_batch.shape[0]
        batch_points = points.reshape(batch_shape + (-1, 3))
        batch_points[:, :, :, :2] *= im_info[:, None, None, 2:4]
        batch_points = batch_points.reshape(points.shape)

        # Predict tokens and masks.
        inputs = self.model.get_inputs({"img": im_batch})
        inputs.update(self.model.get_features(inputs))
        outputs = self.model.get_outputs(dict(**inputs, **{"points": batch_points}))

        # Select final mask.
        iou_pred = outputs["iou_pred"].cpu().numpy()
        point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
        rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
        mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
        iou_scores = outputs["iou_pred"][mask_index].cpu().numpy().reshape(batch_shape)

        # Upscale masks to the original image resolution.
        mask_pred = outputs["mask_pred"][mask_index].unsqueeze_(1)
        mask_pred = self.model.upscale_masks(mask_pred, im_batch.shape[1:-1])
        mask_pred = mask_pred.view(batch_shape + mask_pred.shape[2:])

        # Predict concepts.
        concepts, scores = self.model.predict_concept(outputs["sem_embeds"][mask_index])
        concepts, scores = [x.reshape(batch_shape) for x in (concepts, scores)]

        # Generate captions.
        sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
        captions = self.model.generate_text(sem_tokens).reshape(batch_shape)

        # Postprocess results.
        results = []
        for i in range(batch_shape[0]):
            pred_h, pred_w = im_info[i, :2].astype("int")
            masks = mask_pred[i: i + 1, :, :pred_h, :pred_w]
            masks = self.model.upscale_masks(masks, imgs[i].shape[:2]).flatten(0, 1)
            results.append(
                {
                    "scores": np.stack([iou_scores[i], scores[i]], axis=-1),
                    "masks": masks.gt(0).cpu().numpy().astype("uint8"),
                    "concepts": concepts[i],
                    "captions": captions[i],
                }
            )
        return results


class Inference(object):
    """Command to run batched inference."""

    def __init__(self, model_type, weights, concept_weights):
        self.model_type = model_type
        self.weights = weights
        self.concept_weights = concept_weights

        builder = model_registry[self.model_type]
        self.model = builder(checkpoint=self.weights)

        self.predictor = Predictor(self.model, self.concept_weights)

    def run(self, inputs):
        results = self.predictor.get_results(inputs)
        return results
