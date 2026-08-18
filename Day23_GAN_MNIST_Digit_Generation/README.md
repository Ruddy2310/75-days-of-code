# Day 23 — GAN: Generating Handwritten Digits from Noise (NN Experiment #8)
Following Day 20's autoencoder, this project builds its "generative cousin": a
**Generative Adversarial Network (GAN)** that learns to generate realistic
handwritten digits (MNIST) starting from pure random noise.

## How it works

A GAN is two networks trained against each other:

| Network | Job | Analogy |
|---|---|---|
| **Generator (G)** | Takes a random noise vector `z` and turns it into a fake 28×28 digit image | A forger trying to paint fake banknotes |
| **Discriminator (D)** | Looks at an image (real or fake) and predicts how likely it is to be real | A detective trying to spot the forgeries |

Both are trained simultaneously:
- **D** gets better at telling real MNIST digits from G's fakes.
- **G** gets better at fooling D.

Over many rounds of this adversarial game, G is pushed to produce images that
are statistically indistinguishable from real handwritten digits — even
though it never sees a single real image directly; it only ever sees D's
feedback.

This is a **vanilla / fully-connected GAN** (MLPs, not convolutions) — the
simplest version of the idea, chosen so the whole thing trains in minutes on
a CPU. The [Extensions](#extensions--next-steps) section below covers the
natural next step (DCGAN) once you're comfortable with this one.

## Project structure

```
mnist-gan/
├── src/
│   ├── models.py        # Generator & Discriminator network definitions
│   ├── train.py         # Main training loop
│   ├── generate.py      # Load a checkpoint, generate new digits
│   ├── plot_losses.py   # Plot G/D loss curves from training log
│   └── utils.py         # Checkpointing, sample-grid saving, seeding
├── outputs/
│   ├── samples/         # PNG grids of generated digits, saved during training
│   └── checkpoints/     # Saved model weights, one per epoch
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd mnist-gan
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

MNIST is downloaded automatically by `torchvision` the first time you run
training (into `./data/`, ~11 MB).

## Training

```bash
python src/train.py --epochs 50 --batch-size 128 --latent-dim 100
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of passes over the dataset |
| `--batch-size` | 128 | Images per training step |
| `--lr` | 2e-4 | Adam learning rate (for both G and D) |
| `--latent-dim` | 100 | Size of the random noise vector `z` |
| `--sample-interval` | 200 | Save a sample image grid every N batches |
| `--cpu` | off | Force CPU even if a GPU is available |

While training runs you'll see loss values printed to the console, and PNG
grids of generated digits will accumulate in `outputs/samples/` so you can
watch the digits sharpen over time. A checkpoint is saved after every epoch
to `outputs/checkpoints/`.

**What to expect:** with the default MLP architecture, recognizable (if a
bit blurry) digits typically emerge within 15–20 epochs, and results
continue to improve up to ~50 epochs.

## Generating new digits from a trained model

```bash
python src/generate.py --checkpoint outputs/checkpoints/gan_epoch050.pt \
    --num-images 64 --output outputs/final_samples.png
```

## Plotting training curves

```bash
python src/plot_losses.py --log outputs/loss_log.csv --output outputs/loss_curve.png
```

In a healthy GAN training run, both losses hover and oscillate around
similar values (often near `ln(2) ≈ 0.69` early on) rather than one
collapsing to zero — that's a sign D is overpowering G (or vice versa) and
training has stalled ("mode collapse" or vanishing gradients).

## Results

*(Add your own generated sample grid and loss curve here once you've
trained the model — e.g.)*

```
outputs/samples/epoch050_batch0400.png   <- grid of generated digits
outputs/loss_curve.png                   <- G/D loss over training
```

## From autoencoders to GANs — the conceptual link

Day 20's autoencoder learned to **compress and reconstruct** real images
through a bottleneck — it needs a real image as input to produce an output.
A GAN's generator instead learns to **synthesize** an image from nothing but
random noise, with no direct access to real data at all — it only learns
indirectly, through the discriminator's judgments. That shift, from
*reconstruction* to *adversarial synthesis*, is what makes GANs a natural
"generative cousin" of autoencoders, and also what makes them trickier to
train (there's no simple reconstruction loss to fall back on — just two
networks pulling against each other).

## Extensions / next steps

- **DCGAN**: swap the `Linear` layers for `Conv2d` / `ConvTranspose2d` —
  usually sharper images, especially past epoch 30.
- **Conditional GAN (cGAN)**: feed the digit label into both G and D so you
  can request *"generate a 7"* instead of a random digit.
- **Label smoothing / noisy labels**: use `0.9` instead of `1.0` for "real"
  targets to stabilize training.
- **Wasserstein GAN (WGAN-GP)**: replaces BCE loss with an approximation of
  Earth-Mover distance — much more stable training, fewer mode-collapse
  issues.

## License

MIT — see [LICENSE](LICENSE).
