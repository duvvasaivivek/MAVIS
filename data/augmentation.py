"""
Data Augmentation Module
=========================
TensorFlow-based augmentation functions for Tiny ImageNet (64×64×3).
Augmentation must be IDENTICAL across all four core models for fair comparison.
"""

import tensorflow as tf
from typing import Tuple


def create_augmentation_fn(
    image_size: int = 64,
    random_flip: bool = True,
    random_crop_padding: int = 4,
    color_jitter: bool = True,
    color_jitter_factor: float = 0.2,
    seed: int = 42,
) -> callable:
    """
    Create a training augmentation function.
    
    Applied augmentations (per spec §5.3):
        - Random horizontal flip
        - Random crop with padding
        - Moderate color perturbation
    
    Args:
        image_size: Target image size (64 for Tiny ImageNet).
        random_flip: Enable random horizontal flip.
        random_crop_padding: Pixels of padding before random crop.
        color_jitter: Enable color jitter.
        color_jitter_factor: Strength of color jitter.
        seed: Random seed for reproducibility.
    
    Returns:
        A function that takes (image, label) and returns augmented (image, label).
    """

    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply training augmentation to a single image."""
        # Random horizontal flip
        if random_flip:
            image = tf.image.random_flip_left_right(image, seed=seed)

        # Random crop with padding
        if random_crop_padding > 0:
            padded_size = image_size + 2 * random_crop_padding
            image = tf.image.resize_with_crop_or_pad(image, padded_size, padded_size)
            image = tf.image.random_crop(
                image, [image_size, image_size, 3], seed=seed
            )

        # Color jitter (brightness, contrast, saturation)
        if color_jitter:
            image = tf.image.random_brightness(image, max_delta=color_jitter_factor, seed=seed)
            image = tf.image.random_contrast(
                image, lower=1.0 - color_jitter_factor, 
                upper=1.0 + color_jitter_factor, seed=seed
            )
            image = tf.image.random_saturation(
                image, lower=1.0 - color_jitter_factor, 
                upper=1.0 + color_jitter_factor, seed=seed
            )
            # Clip to valid range after jitter
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    return augment


def normalize_image(
    image: tf.Tensor, label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0].
    
    Args:
        image: uint8 image tensor.
        label: Integer label.
    
    Returns:
        (normalized_image, label)
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_validation_fn(image_size: int = 64) -> callable:
    """
    Create a validation/test preprocessing function.
    
    No augmentation — only normalization and optional center crop.
    
    Args:
        image_size: Target image size.
    
    Returns:
        A function that takes (image, label) and returns preprocessed (image, label).
    """

    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Center crop and normalize for evaluation."""
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        return image, label

    return preprocess
