import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from distributed_kron import kron


init_fn = lambda dim: nn.initializers.normal(jnp.sqrt(2 / (5 * dim)))
wang_fn = lambda dim, n_layers: nn.initializers.normal(2 / n_layers / jnp.sqrt(dim))


class RMSNorm(nn.Module):
    reduction_axes: tuple[int, ...] = -1

    @nn.compact
    def __call__(self, x):
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=self.reduction_axes, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)
        return normed_inputs.astype(x.dtype)


def _dot_product_attention_core(query, key, value):
    head_dim = query.shape[-1]
    query *= jax.lax.rsqrt(jnp.array(head_dim, dtype=jnp.float32)).astype(query.dtype)
    logits = jnp.einsum("BTNH,BSNH->BNTS", query, key)
    probs = jax.nn.softmax(logits.astype(jnp.float32)).astype(logits.dtype)
    encoded = jnp.einsum("BNTS,BSNH->BTNH", probs, value)
    return encoded


def _sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "i,j->ij",
        jnp.arange(length),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def _apply_rotary_embedding(q, k, cos, sin):
    qlen = q.shape[-4]
    klen = k.shape[-3]
    qcos = jnp.expand_dims(cos[:qlen, :], range(len(q.shape) - 2))
    qsin = jnp.expand_dims(sin[:qlen, :], range(len(q.shape) - 2))
    kcos = jnp.expand_dims(cos[:klen, :], range(len(k.shape) - 2))
    ksin = jnp.expand_dims(sin[:klen, :], range(len(k.shape) - 2))
    qcos = jnp.swapaxes(qcos, -2, -4)
    qsin = jnp.swapaxes(qsin, -2, -4)
    kcos = jnp.swapaxes(kcos, -2, -3)
    ksin = jnp.swapaxes(ksin, -2, -3)
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
        N = self.num_heads
        K = self.num_kv_heads
        G = N // K
        H = C // N

        q_params = self.param("q_kernel", init_fn(C), (C, N * H))
        k_params = self.param("k_kernel", init_fn(C), (C, K * H))
        v_params = self.param("v_kernel", init_fn(C), (C, K * H))
        out_params = self.param("out_kernel", wang_fn(N * H, self.n_layers), (N * H, C))

        q = jnp.dot(x, q_params)
        k = jnp.dot(x, k_params)
        v = jnp.dot(x, v_params)

        q = RMSNorm()(q)
        k = RMSNorm()(k)

        q = jnp.reshape(q, (B, T, K, G, H))
        k = jnp.reshape(k, (B, T, K, H))
        v = jnp.reshape(v, (B, T, K, H))

        sin, cos = _sine_table(H, T, max_timescale=10000.0)
        q, k = _apply_rotary_embedding(q, k, cos, sin)

        vmapped_fn = jax.vmap(
            _dot_product_attention_core, in_axes=(3, None, None), out_axes=3
        )
        encoded = vmapped_fn(q, k, v)
        encoded = jnp.reshape(encoded, (B, T, N * H))
        out = jnp.dot(encoded, out_params)
        return out


class MLP(nn.Module):
    n_layers: int

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]
        hid = C * 2
        up_kernel = self.param("up_kernel", init_fn(C), (C, hid))
        down_kernel = self.param("down_kernel", wang_fn(hid, self.n_layers), (hid, C))
        x = jnp.dot(x, up_kernel)
        x = nn.silu(x)
        return jnp.dot(x, down_kernel)


class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        attn_layer = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.n_layers,
        )
        x += attn_layer(RMSNorm()(x))
        x += MLP(self.n_layers)(RMSNorm()(x))
        return x


class VisionTransformer(nn.Module):
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    num_classes: int
    patch_size: int = 4  # Changed to 4 for CIFAR-10 (32x32 images)

    @nn.compact
    def __call__(self, x, train: bool = True):
        B, H, W, C = x.shape
        
        # Patch embedding
        x = nn.Conv(self.embed_dim, kernel_size=(self.patch_size, self.patch_size), 
                   strides=(self.patch_size, self.patch_size))(x)
        
        # Reshape to sequence
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        
        # Add positional embedding
        pos_embedding = self.param("pos_embedding", 
                                  nn.initializers.normal(0.02), 
                                  (1, x.shape[1], self.embed_dim))
        x = x + pos_embedding
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = Block(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                n_layers=self.num_layers,
            )(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)
        
        # Layer norm before classifier
        x = RMSNorm()(x)
        
        # Classifier
        x = nn.Dense(self.num_classes)(x)
        
        return x


def create_train_state(rng, model, learning_rate, weight_decay):
    dummy_input = jnp.ones((2, 32, 32, 3))
    params = model.init(rng, dummy_input)
    
    optimizer = kron(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.mean(loss)


@jax.jit
def train_step(state, batch):
    images, labels = batch
    
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
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


def random_crop(image, padding=4):
    """Randomly crop the image with padding."""
    height, width = image.shape[:2]
    padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    # Random crop coordinates
    h_start = np.random.randint(0, 2 * padding + 1)
    w_start = np.random.randint(0, 2 * padding + 1)
    
    return padded[h_start:h_start+height, w_start:w_start+width, :]


def random_flip(image):
    """Randomly flip the image horizontally."""
    if np.random.random() < 0.5:
        return image[:, ::-1, :]
    return image


def color_jitter(image, brightness=0.1, contrast=0.1, saturation=0.1):
    """Apply random color jittering to the image."""
    # Brightness adjustment
    if np.random.random() < 0.5:
        factor = 1.0 + np.random.uniform(-brightness, brightness)
        image = np.clip(image * factor, 0, 1)
    
    # Contrast adjustment
    if np.random.random() < 0.5:
        factor = 1.0 + np.random.uniform(-contrast, contrast)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * factor + mean, 0, 1)
    
    # Simple saturation adjustment (approximation)
    if np.random.random() < 0.5:
        factor = 1.0 + np.random.uniform(-saturation, saturation)
        gray = np.mean(image, axis=2, keepdims=True)
        image = np.clip(gray + factor * (image - gray), 0, 1)
    
    return image


def prepare_data(batch_size=128):
    """Prepare CIFAR-10 data for training and evaluation."""
    print("Loading CIFAR-10 dataset...")
    dataset = load_dataset("cifar10")
    
    # Convert to numpy format
    train_dataset = dataset["train"].with_format("numpy")
    test_dataset = dataset["test"].with_format("numpy")
    
    # Pre-process all images once to avoid repeated processing
    print("Pre-processing training data...")
    train_images = np.array(train_dataset["img"], dtype=np.float32) / 255.0
    train_labels = np.array(train_dataset["label"], dtype=np.int32)
    
    print("Pre-processing test data...")
    test_images = np.array(test_dataset["img"], dtype=np.float32) / 255.0
    test_labels = np.array(test_dataset["label"], dtype=np.int32)
    
    # Create data generators
    def train_generator():
        num_samples = len(train_images)
        indices = np.random.permutation(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Apply data augmentation
            batch_images = []
            for idx in batch_indices:
                img = train_images[idx].copy()
                img = random_crop(img)
                img = random_flip(img)
                img = color_jitter(img)
                batch_images.append(img)
            
            batch_images = np.stack(batch_images)
            batch_labels = train_labels[batch_indices]
            
            yield batch_images, batch_labels
    
    def test_generator():
        num_samples = len(test_images)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_images = test_images[start_idx:end_idx]
            batch_labels = test_labels[start_idx:end_idx]
            
            yield batch_images, batch_labels
    
    return train_generator, test_generator


def train_model(model_config, train_config):
    """Train the Vision Transformer model on CIFAR-10."""
    print("Initializing model...")
    rng = jax.random.PRNGKey(train_config["seed"])
    rng, init_rng = jax.random.split(rng)
    
    model = VisionTransformer(**model_config)
    state = create_train_state(
        init_rng, 
        model, 
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"]
    )
    
    train_generator, test_generator = prepare_data(train_config["batch_size"])
    
    print_every = train_config.get("print_every", 50)
    step = 0
    
    for epoch in range(train_config["num_epochs"]):
        print(f"Epoch {epoch+1}/{train_config['num_epochs']}")
        
        # Training
        train_losses = []
        train_accuracies = []
        
        for batch in train_generator():
            state, loss, accuracy = train_step(state, batch)
            train_losses.append(loss)
            train_accuracies.append(accuracy)
            step += 1
            
            if step % print_every == 0:
                avg_loss = np.mean(train_losses[-print_every:])
                avg_acc = np.mean(train_accuracies[-print_every:])
                print(f"  Step {step}: Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
        
        # Evaluation
        eval_losses = []
        eval_accuracies = []
        
        for batch in test_generator():
            loss, accuracy = eval_step(state, batch)
            eval_losses.append(loss)
            eval_accuracies.append(accuracy)
        
        avg_eval_loss = float(np.mean(eval_losses))
        avg_eval_accuracy = float(np.mean(eval_accuracies))
        
        print(f"  Epoch {epoch+1} Evaluation - Loss: {avg_eval_loss:.4f}, Accuracy: {avg_eval_accuracy:.4f}")
    
    return state


if __name__ == "__main__":
    model_config = {
        "embed_dim": 192,
        "num_heads": 6,
        "num_kv_heads": 3,
        "num_layers": 6,
        "num_classes": 10,
        "patch_size": 4,  # Use smaller patches for CIFAR-10
    }
    
    train_config = {
        "batch_size": 128,
        "learning_rate": 0.0001,  # Adjusted learning rate
        "weight_decay": 0.1,      # Adjusted weight decay
        "num_epochs": 10,
        "seed": 42,
        "print_every": 50,
    }
    
    train_model(model_config, train_config)
