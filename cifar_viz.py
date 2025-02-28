import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from datasets import load_dataset
from distributed_kron import kron

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

init_fn = lambda dim: nn.initializers.normal(jnp.sqrt(2 / (5 * dim)))
wang_fn = lambda dim, n_layers: nn.initializers.normal(2 / n_layers / jnp.sqrt(dim))


class RMSNorm(nn.Module):
    reduction_axes: tuple[int, ...] = -1

    @nn.compact
    def __call__(self, x):
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=self.reduction_axes, keepdims=True)
        return (x * jax.lax.rsqrt(var + 1e-06)).astype(x.dtype)


def _dot_product_attention_core(query, key, value):
    head_dim = query.shape[-1]
    query *= jax.lax.rsqrt(jnp.array(head_dim, dtype=jnp.float32)).astype(query.dtype)
    logits = jnp.einsum("BTNH,BSNH->BNTS", query, key)
    probs = jax.nn.softmax(logits.astype(jnp.float32)).astype(logits.dtype)
    return jnp.einsum("BNTS,BSNH->BTNH", probs, value)


def _sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(length), 1.0 / timescale, 
                             precision=jax.lax.Precision.HIGHEST)
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rotary_embedding(q, k, cos, sin):
    qlen, klen = q.shape[-4], k.shape[-3]
    qcos = jnp.expand_dims(cos[:qlen, :], range(len(q.shape) - 2))
    qsin = jnp.expand_dims(sin[:qlen, :], range(len(q.shape) - 2))
    kcos = jnp.expand_dims(cos[:klen, :], range(len(k.shape) - 2))
    ksin = jnp.expand_dims(sin[:klen, :], range(len(k.shape) - 2))
    
    qcos, qsin = jnp.swapaxes(qcos, -2, -4), jnp.swapaxes(qsin, -2, -4)
    kcos, ksin = jnp.swapaxes(kcos, -2, -3), jnp.swapaxes(ksin, -2, -3)
    
    out_q = q * qcos + _rotate_half(q) * qsin
    out_k = k * kcos + _rotate_half(k) * ksin
    return out_q.astype(q.dtype), out_k.astype(k.dtype)


class Attention(nn.Module):
    num_heads: int
    num_kv_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        N, K = self.num_heads, self.num_kv_heads
        G, H = N // K, C // N

        q_params = self.param("q_kernel", init_fn(C), (C, N * H))
        k_params = self.param("k_kernel", init_fn(C), (C, K * H))
        v_params = self.param("v_kernel", init_fn(C), (C, K * H))
        out_params = self.param("out_kernel", wang_fn(N * H, self.n_layers), (N * H, C))

        q, k, v = jnp.dot(x, q_params), jnp.dot(x, k_params), jnp.dot(x, v_params)
        q, k = RMSNorm()(q), RMSNorm()(k)
        q = jnp.reshape(q, (B, T, K, G, H))
        k, v = jnp.reshape(k, (B, T, K, H)), jnp.reshape(v, (B, T, K, H))

        sin, cos = _sine_table(H, T, max_timescale=10000.0)
        q, k = _apply_rotary_embedding(q, k, cos, sin)

        encoded = jax.vmap(_dot_product_attention_core, in_axes=(3, None, None), out_axes=3)(q, k, v)
        return jnp.dot(jnp.reshape(encoded, (B, T, N * H)), out_params)


class MLP(nn.Module):
    n_layers: int

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]
        hid = C * 2
        up_kernel = self.param("up_kernel", init_fn(C), (C, hid))
        down_kernel = self.param("down_kernel", wang_fn(hid, self.n_layers), (hid, C))
        return jnp.dot(nn.silu(jnp.dot(x, up_kernel)), down_kernel)


class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x += Attention(self.num_heads, self.num_kv_heads, self.n_layers)(RMSNorm()(x))
        x += MLP(self.n_layers)(RMSNorm()(x))
        return x


class VisionTransformer(nn.Module):
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    num_classes: int
    patch_size: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        B, H, W, C = x.shape
        x = nn.Conv(self.embed_dim, kernel_size=(self.patch_size, self.patch_size), 
                   strides=(self.patch_size, self.patch_size))(x)
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        
        pos_embedding = self.param("pos_embedding", nn.initializers.normal(0.02), 
                                  (1, x.shape[1], self.embed_dim))
        x = x + pos_embedding
        
        for _ in range(self.num_layers):
            x = Block(self.num_heads, self.num_kv_heads, self.num_layers)(x)
        
        return nn.Dense(self.num_classes)(RMSNorm()(jnp.mean(x, axis=1)))


def create_train_state(rng, model, learning_rate, weight_decay, total_steps):
    params = model.init(rng, jnp.ones((2, 32, 32, 3)))
    lr_schedule = optax.linear_schedule(learning_rate, 1e-8, total_steps)
    optimizer = kron(
        learning_rate=lr_schedule, weight_decay=weight_decay,
        merge_small_dims=False, partition_grads_into_blocks=False,
        preconditioner_update_probability=1.0,  # run every step for better visualization
        preconditioner_lr=1.0,
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return jnp.mean(-jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))


@jax.jit
def train_step(state, batch):
    images, labels = batch
    
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        return cross_entropy_loss(logits, labels), logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    
    return state, loss, accuracy


@jax.jit
def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn(state.params, images)
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy


def random_crop(images, padding=4):
    batch_size, height, width, channels = images.shape
    padded = np.pad(images, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h_starts = np.random.randint(0, 2 * padding + 1, size=batch_size)
    w_starts = np.random.randint(0, 2 * padding + 1, size=batch_size)
    
    cropped_images = np.zeros_like(images)
    for i, (h_start, w_start) in enumerate(zip(h_starts, w_starts)):
        cropped_images[i] = padded[i, h_start:h_start+height, w_start:w_start+width, :]
    
    return cropped_images


def random_flip(images):
    flip_mask = np.random.random(size=images.shape[0]) < 0.5
    flipped_images = images.copy()
    flipped_images[flip_mask] = flipped_images[flip_mask, :, ::-1, :]
    return flipped_images


def color_jitter(images, brightness=0.1, contrast=0.1, saturation=0.1):
    batch_size = images.shape[0]
    jittered_images = images.copy()
    
    brightness_factors = np.random.uniform(1-brightness, 1+brightness, size=batch_size)
    brightness_mask = np.random.random(size=batch_size) < 0.5
    for i, (factor, apply) in enumerate(zip(brightness_factors, brightness_mask)):
        if apply:
            jittered_images[i] = np.clip(jittered_images[i] * factor, 0, 1)
    
    contrast_factors = np.random.uniform(1-contrast, 1+contrast, size=batch_size)
    contrast_mask = np.random.random(size=batch_size) < 0.5
    for i, (factor, apply) in enumerate(zip(contrast_factors, contrast_mask)):
        if apply:
            mean = np.mean(jittered_images[i], axis=(0, 1), keepdims=True)
            jittered_images[i] = np.clip((jittered_images[i] - mean) * factor + mean, 0, 1)
    
    saturation_factors = np.random.uniform(1-saturation, 1+saturation, size=batch_size)
    saturation_mask = np.random.random(size=batch_size) < 0.5
    for i, (factor, apply) in enumerate(zip(saturation_factors, saturation_mask)):
        if apply:
            gray = np.mean(jittered_images[i], axis=2, keepdims=True)
            jittered_images[i] = np.clip(gray + factor * (jittered_images[i] - gray), 0, 1)
    
    return jittered_images


def normalize_images(images):
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 1, 1, 3)
    return (images - mean) / std


def prepare_data(batch_size=128):
    dataset = load_dataset("cifar10")
    train_dataset = dataset["train"].with_format("numpy")
    test_dataset = dataset["test"].with_format("numpy")
    
    train_images = np.array(train_dataset["img"], dtype=np.float32) / 255.0
    train_labels = np.array(train_dataset["label"], dtype=np.int32)
    test_images = np.array(test_dataset["img"], dtype=np.float32) / 255.0
    test_labels = np.array(test_dataset["label"], dtype=np.int32)
    
    def train_generator():
        indices = np.random.permutation(len(train_images))
        for start_idx in range(0, len(train_images), batch_size):
            end_idx = min(start_idx + batch_size, len(train_images))
            batch_indices = indices[start_idx:end_idx]
            batch_images = train_images[batch_indices].copy()
            yield normalize_images(color_jitter(random_flip(random_crop(batch_images)))), train_labels[batch_indices]
    
    def test_generator():
        for start_idx in range(0, len(test_images), batch_size):
            end_idx = min(start_idx + batch_size, len(test_images))
            yield normalize_images(test_images[start_idx:end_idx].copy()), test_labels[start_idx:end_idx]
    
    return train_generator, test_generator


def visualize_kron_optimizer(state, epoch, save_dir="assets"):
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer_state = state.opt_state
    params = state.params
    
    param_path = ["params", "Block_2", "Attention_0", "q_kernel"]
    display_name = "Block 2 - Attention Query"
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Gradient Whitening with PSGD Kron - {display_name}", fontsize=16)
    
    titles = [
        ["Gradient", "Gradient Gram", "Left Q", "Right Q"],
        ["Whitened Gradient", "Whitened Gradient Gram", "Left Preconditioner", "Right Preconditioner"]
    ]
    
    for row in range(2):
        for col, title in enumerate(titles[row]):
            axes[row, col].set_title(title)
    
    preconditioners = optimizer_state[0].get('Qs_preconditioners', {})
    all_grads = optimizer_state[0].get('mu', {})
    
    param = params
    for k in param_path:
        param = param[k]
    
    grad = all_grads
    for k in param_path:
        grad = grad[k]
    
    current_precond = preconditioners
    for k in param_path[:-1]:
        if k in current_precond:
            current_precond = current_precond[k]
    
    precond_factors = current_precond[param_path[-1]]
    left_q, right_q = precond_factors[0], precond_factors[1]
    
    grad_gram = jnp.matmul(grad, grad.T)
    left_p = jnp.matmul(left_q.T, left_q)
    right_p = jnp.matmul(right_q.T, right_q)
    whitened_grad = jnp.matmul(jnp.matmul(left_p, grad), right_p.T)
    whitened_gram = jnp.matmul(whitened_grad, whitened_grad.T)
    
    matrices = [
        [grad, grad_gram, left_q, right_q],
        [whitened_grad, whitened_gram, left_p, right_p]
    ]

    cmap = 'hot'
    
    for row in range(2):
        for col in range(4):
            matrix = matrices[row][col]
            vmin, vmax = None, None
            if col == 0:
                vmin, vmax = -np.abs(matrix).max(), np.abs(matrix).max()
            im = axes[row, col].imshow(
                np.array(matrix),
                cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal'
            )
            plt.colorbar(im, ax=axes[row, col])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"kron_visualization_epoch_{epoch}.png"), dpi=300)
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Gradient Orthogonalization with SVD - {display_name}", fontsize=16)
    
    u, _, vh = jnp.linalg.svd(grad, full_matrices=False)
    whitened_grad_svd = jnp.matmul(u, vh)
    whitened_gram_svd = jnp.matmul(whitened_grad_svd, whitened_grad_svd.T)

    titles = [["Original Gradient", "Original Gradient Gram"], 
              ["Orthogonalized Gradient", "Orthogonalized Gradient Gram"]]
    
    matrices = [[grad, grad_gram], 
                [whitened_grad_svd, whitened_gram_svd]]
    
    for row in range(2):
        for col in range(2):
            matrix = matrices[row][col]
            vmin, vmax = None, None
            if col == 0 or row == 0:
                vmin, vmax = -np.abs(matrix).max(), np.abs(matrix).max()
            im = axes[row, col].imshow(
                np.array(matrix),
                cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal'
            )
            plt.colorbar(im, ax=axes[row, col])
            axes[row, col].set_title(titles[row][col])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"ortho_visualization_epoch_{epoch}.png"), dpi=300)
    plt.close()


def train_model(model_config, train_config):
    rng = jax.random.PRNGKey(train_config["seed"])
    rng, init_rng = jax.random.split(rng)
    model = VisionTransformer(**model_config)
    
    train_generator, test_generator = prepare_data(train_config["batch_size"])
    total_steps = train_config["num_epochs"] * 50000 // train_config["batch_size"]
    
    state = create_train_state(
        init_rng, model, train_config["learning_rate"],
        train_config["weight_decay"], total_steps
    )

    print_every = train_config.get("print_every", 50)
    step = 0
    
    for epoch in range(train_config["num_epochs"]):
        print(f"Epoch {epoch+1}/{train_config['num_epochs']}")
        
        train_losses, train_accuracies = [], []
        for batch in train_generator():
            state, loss, accuracy = train_step(state, batch)
            train_losses.append(loss)
            train_accuracies.append(accuracy)
            step += 1
            
            if step % print_every == 0:
                avg_loss = np.mean(train_losses[-print_every:])
                avg_acc = np.mean(train_accuracies[-print_every:])
                print(f"  Step {step}: Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
        
        eval_losses, eval_accuracies = [], []
        for batch in test_generator():
            loss, accuracy = eval_step(state, batch)
            eval_losses.append(loss)
            eval_accuracies.append(accuracy)
        
        avg_eval_loss = float(np.mean(eval_losses))
        avg_eval_accuracy = float(np.mean(eval_accuracies))
        print(f"  Epoch {epoch+1} Evaluation - Loss: {avg_eval_loss:.4f}, "
              f"Accuracy: {avg_eval_accuracy:.4f}")
        
        if epoch + 1 == train_config["num_epochs"]:
            visualize_kron_optimizer(state, epoch + 1)
    
    return state


if __name__ == "__main__":
    model_config = {
        "embed_dim": 192, "num_heads": 6, "num_kv_heads": 3,
        "num_layers": 6, "num_classes": 10, "patch_size": 4,
    }
    
    train_config = {
        "batch_size": 128, "learning_rate": 0.003, "weight_decay": 0.7,
        "num_epochs": 4, "seed": 42, "print_every": 50,
    }
    
    train_model(model_config, train_config)
