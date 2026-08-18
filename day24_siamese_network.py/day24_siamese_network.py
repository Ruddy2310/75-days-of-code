"""
Day 24 - Deep Learning Basics: NN Experiment #9
Siamese Network for Similarity Learning (Contrastive Loss)

Part of #75DaysOfCode.

Two identical sub-networks (shared weights) each embed an input into a
low-dimensional feature vector. Contrastive loss pulls embeddings of
same-class pairs together and pushes different-class pairs apart, so the
network learns a *similarity metric* instead of a direct classifier -
this is the same idea used in face/signature verification systems.

Everything below (data generation, forward pass, backward pass, training
loop) is implemented from scratch in numpy - no framework, no pretrained
weights, no internet access required.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------------------------------
# 1. Synthetic dataset: two visually distinct 8x8 "image" classes
#    Class 0 = filled circle, Class 1 = filled square (with noise)
# --------------------------------------------------------------------------
IMG_SIZE = 8
FLAT_DIM = IMG_SIZE * IMG_SIZE


def make_circle(noise=0.15):
    img = np.zeros((IMG_SIZE, IMG_SIZE))
    cx, cy, r = 3.5, 3.5, 2.6
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                img[i, j] = 1.0
    img += np.random.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).flatten()


def make_square(noise=0.15):
    img = np.zeros((IMG_SIZE, IMG_SIZE))
    img[1:7, 1:7] = 1.0
    img += np.random.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).flatten()


def make_dataset(n_per_class=150):
    X, y = [], []
    for _ in range(n_per_class):
        X.append(make_circle())
        y.append(0)
    for _ in range(n_per_class):
        X.append(make_square())
        y.append(1)
    X = np.array(X)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def make_pairs(X, y, n_pairs=2000):
    """Sample balanced positive (same class) and negative (diff class) pairs."""
    pairs_a, pairs_b, labels = [], [], []
    n = len(X)
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]

    for _ in range(n_pairs // 2):
        # positive pair (same class) -> label 1
        cls_idx = class0_idx if np.random.rand() < 0.5 else class1_idx
        i, j = np.random.choice(cls_idx, 2, replace=True)
        pairs_a.append(X[i]); pairs_b.append(X[j]); labels.append(1)

        # negative pair (different class) -> label 0
        i = np.random.choice(class0_idx)
        j = np.random.choice(class1_idx)
        pairs_a.append(X[i]); pairs_b.append(X[j]); labels.append(0)

    return np.array(pairs_a), np.array(pairs_b), np.array(labels)


# --------------------------------------------------------------------------
# 2. Shared-weight twin encoder (manual numpy MLP)
#    64 -> 32 -> 16 -> 2  (embedding dim = 2, so we can plot it directly)
# --------------------------------------------------------------------------
class SiameseEncoder:
    def __init__(self, in_dim=64, h1=32, h2=16, emb_dim=2, lr=0.05):
        self.lr = lr
        self.W1 = np.random.randn(in_dim, h1) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, emb_dim) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(emb_dim)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        emb = a2 @ self.W3 + self.b3
        cache = (x, z1, a1, z2, a2, emb)
        return emb, cache

    def backward(self, d_emb, cache):
        x, z1, a1, z2, a2, emb = cache
        dW3 = a2.T @ d_emb
        db3 = d_emb.sum(axis=0)
        da2 = d_emb @ self.W3.T
        dz2 = da2 * self.relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_grad(z1)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2, dW3, db3

    def step(self, grads, batch_size):
        dW1, db1, dW2, db2, dW3, db3 = grads
        self.W1 -= self.lr * dW1 / batch_size
        self.b1 -= self.lr * db1 / batch_size
        self.W2 -= self.lr * dW2 / batch_size
        self.b2 -= self.lr * db2 / batch_size
        self.W3 -= self.lr * dW3 / batch_size
        self.b3 -= self.lr * db3 / batch_size


def contrastive_loss_and_grad(e1, e2, y, margin=1.0):
    """
    y = 1 -> same class (pull together): loss = D^2
    y = 0 -> diff class (push apart):    loss = max(0, margin - D)^2
    Returns mean loss and gradients w.r.t. e1 and e2.
    """
    diff = e1 - e2
    D = np.sqrt((diff ** 2).sum(axis=1) + 1e-9)  # (batch,)

    same_loss = D ** 2
    diff_loss = np.clip(margin - D, 0, None) ** 2
    loss = np.where(y == 1, same_loss, diff_loss)

    # dLoss/dD
    dD = np.where(y == 1, 2 * D, -2 * np.clip(margin - D, 0, None))
    # dD/d(diff) = diff / D
    dD_ddiff = diff / D[:, None]
    d_diff = (dD[:, None]) * dD_ddiff

    d_e1 = d_diff
    d_e2 = -d_diff
    return loss.mean(), d_e1, d_e2, D


# --------------------------------------------------------------------------
# 3. Train
# --------------------------------------------------------------------------
def train(epochs=400, batch_size=64, lr=0.05, margin=1.0):
    X, y = make_dataset(n_per_class=150)
    Xa, Xb, pair_y = make_pairs(X, y, n_pairs=2000)

    net = SiameseEncoder(in_dim=FLAT_DIM, lr=lr)
    n = len(pair_y)
    losses = []

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            xa, xb, yb = Xa[b_idx], Xb[b_idx], pair_y[b_idx]

            e1, cache1 = net.forward(xa)
            e2, cache2 = net.forward(xb)
            loss, d_e1, d_e2, D = contrastive_loss_and_grad(e1, e2, yb, margin)

            grads1 = net.backward(d_e1, cache1)
            grads2 = net.backward(d_e2, cache2)
            # shared weights -> accumulate gradients from both branches
            combined = [g1 + g2 for g1, g2 in zip(grads1, grads2)]
            net.step(combined, batch_size=len(b_idx))

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | contrastive loss: {avg_loss:.4f}")

    return net, losses, X, y, Xa, Xb, pair_y


if __name__ == "__main__":
    net, losses, X, y, Xa, Xb, pair_y = train(epochs=400, batch_size=64, lr=0.05, margin=1.0)

    # ---- Plot 1: training loss curve ----
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="#2b6cb0")
    plt.title("Siamese Network - Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("siamese_training_curve.png", dpi=130)
    plt.close()

    # ---- Plot 2: learned embedding space ----
    emb, _ = net.forward(X)
    plt.figure(figsize=(6, 6))
    plt.scatter(emb[y == 0, 0], emb[y == 0, 1], c="#e53e3e", label="circle", alpha=0.7)
    plt.scatter(emb[y == 1, 0], emb[y == 1, 1], c="#3182ce", label="square", alpha=0.7)
    plt.title("Learned Embedding Space (2D)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("embedding_space_visualization.png", dpi=130)
    plt.close()

    # ---- Plot 3: example pairs with predicted distance ----
    n_show = 6
    show_idx = np.random.choice(len(pair_y), n_show, replace=False)
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4.5))
    for col, i in enumerate(show_idx):
        a_img = Xa[i].reshape(IMG_SIZE, IMG_SIZE)
        b_img = Xb[i].reshape(IMG_SIZE, IMG_SIZE)
        ea, _ = net.forward(Xa[i:i + 1])
        eb, _ = net.forward(Xb[i:i + 1])
        dist = np.sqrt(((ea - eb) ** 2).sum())
        label = "SAME" if pair_y[i] == 1 else "DIFF"

        axes[0, col].imshow(a_img, cmap="gray")
        axes[0, col].axis("off")
        axes[1, col].imshow(b_img, cmap="gray")
        axes[1, col].axis("off")
        axes[0, col].set_title(f"{label}\nD={dist:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig("similarity_pairs_comparison.png", dpi=130)
    plt.close()

    print("\nDone. Saved:")
    print(" - siamese_training_curve.png")
    print(" - embedding_space_visualization.png")
    print(" - similarity_pairs_comparison.png")
