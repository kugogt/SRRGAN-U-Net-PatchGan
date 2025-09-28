# 🧠 SRRGAN: Super-Resolution and Restoration with a U-Net Generator and PatchGAN Discriminator

Ciao! 👋 Welcome to this personal Computer Vision project, an evolution of [my previous work on image super-resolution]([https://example.com](https://github.com/kugogt/SR-Restoration-images-U-Net/tree/main)). This work implements a complete image-to-image pipeline, where a U-Net-based generator is trained adversarially against a PatchGAN discriminator. The primary goal is to move beyond the limitations of traditional pixel-wise losses (like MAE) and generate images that are not only accurate but also perceptually sharp and realistic.

Training was done on the DK2K dataset and the test evaluation on BSDS100 dataset.

---

## 🧩 Project Highlights

This project was engineered with several key features:

- **Generative Adversarial Network (GAN)**: A U-Net generator is paired with a PatchGAN discriminator, which evaluates image realism on overlapping patches. This encourages the generator to produce fine-grained, authentic textures.
- **"Blind" Degradation Pipeline**: To train a model for real-world scenarios, a degradation function was designed. It applies stochastic (randomized) degradations during training and deterministic (fixed) degradations during validation, creating a "blind restoration" task.
- **U-Net Generator**: The generator's architecture is augmented with Squeeze-and-Excitation (SE) and Spatial Attention blocks, PixelShuffle upsampling, and Global Residual Learning.
- **Three-Stage Training**: To manage the instability associated with GANs, a three-phase training strategy was employed. This ensures the generator reaches a strong starting point before engaging in adversarial training.
- **Perceptual-Driven Model Selection**: Recognizing that metrics like PSNR do not always align with human vision, the best-performing generator was selected by evaluating all saved checkpoints with the LPIPS (Learned Perceptual Image Patch Similarity) metric.
- **Seamless Tiled Inference**: An inference pipeline was developed to handle images of any size. It uses overlapping patches and Hann window blending to produce a seamless output.

---

## 📦 The Degradation Model

A key component of this project is the pipeline used to create low-resolution (LR) images from their high-resolution (HR) counterparts. This process is intentionally different for training and evaluation.

- **For Training (Stochastic)**:  The model is exposed to a wide variety of artifacts by randomizing the degradation parameters and order. This includes:
  - **Blur**: Random choice between Gaussian and Box blur.
  - **Downsampling (2x)**: Random choice of interpolation method (Bilinear, Bicubic, Area)
  - **Noise & Compression**: A randomized sequence of Gaussian/Poisson noise and JPEG compression with random quality.
- **For Validation/Testing (Deterministic)**: A fixed degradation pipeline is used to ensure consistent and reproducible performance measurement across all evaluation steps.

---

## ⚙️ Model Architectures

1. **The Generator (U-Net)**
   The generator uses a U-Net architecture, leveraging its encoder-decoder structure and skip connections to preserve   spatial information across scales. The core of the network is enhanced with a deep stack of residual blocks and attention mechanisms to refine features and focus on critical image details.
2. **The Discriminator (PatchGAN)**
   The discriminator's role is to distinguish between the generator's output and real HR images. By operating on N×N image patches, it pushes the generator to produce locally realistic details across the entire image. The architecture incorporates Spectral Normalization for training stability and is designed to work with the Least Squares GAN (LSGAN) loss function.

---

## 📉 Training Methodology

A structured, three-phase training approach was used to ensure stability and convergence:

1. **Phase 1: Baseline Pre-training (MAE Loss)**: The generator is trained alone to minimize pixel-wise error, establishing a foundation for image content and structure.
2. **Phase 2: Perceptual Fine-tuning (MAE + VGG Loss)**: The generator is further trained with a composite loss that includes a perceptual component (using VGG19 features), guiding it to produce more visually pleasing results.
3. **Phase 3: Adversarial Training (Full GAN)**: The pre-trained generator and discriminator are trained together. The generator's loss becomes a weighted sum of pixel loss, perceptual loss, and adversarial loss, balancing content fidelity with photorealism.

---

## 🔬 Results & Final Comparison

The SRRGAN model, selected based on the best LPIPS score across all training checkpoints, demostrated the "Perception-Distortion Trade-off":

This trade-off describes the inverse relationship between pixel-level accuracy and perceptual metric:
- **Distortion metrics (PSNR and SSIM)** reward models for being mathematically close to the ground truth, which can favor smoother, less detailed images.
- **Perceptual metrics (like LPIPS)** reward models for creating realistic textures and sharp details, even if those details aren't a perfect pixel-for-pixel match.
As seen in the results below, the U-Net (Perceptual) model achieves the best LPIPS score because it was directly optimized for this. However, the SRRGAN finds a better balance. Its adversarial training generates sharper details that result in the highest PSNR and SSIM scores, while maintaining a better perceptual quality than the Mae training producing images that are both accurate and realistic.

📊 Final Evaluation Results

| Model                                | Test Loss / Combined | Perceptual Loss | MAE    | PSNR   | SSIM  |
|--------------------------------------|----------------------|-----------------|--------|--------|-------|
| **U-Net + MAE**                      | **0.0969**           | –               | 0.0969 | 24.14  | 0.608 |
| **U-Net + Perceptual Loss (α=0.005)** | 0.0995               | 0.9851          | 0.0936 | 24.38  | 0.618 |
| **U-Net + Perceptual Loss (α=0.02)**  | 0.1195               | 0.9856          | 0.0993 | 23.84  | 0.580 |

---

## 📷 Visual Results: Patch-Level detail









