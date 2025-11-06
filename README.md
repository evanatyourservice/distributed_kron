# Distributed PSGD Kron

For original PSGD repo and some great resources, see [psgd_torch](https://github.com/lixilinx/psgd_torch).

**Background**: Implementation of [PSGD Kron](https://github.com/lixilinx/psgd_torch) in JAX (optax-style) for 
distributed training. PSGD is a second-order optimizer originally created by Xi-Lin Li and further developed by
Omead Pooladzandi that uses either a hessian-based or whitening-based (gg^T) preconditioner, lie groups, and
online preconditioner updating to improve training convergence, generalization, and efficiency. I highly suggest
taking a look at Xi-Lin's PSGD repo linked above for interesting details on how PSGD works and experiments using
PSGD. There are also resources listed near the bottom of this readme.

### `distributed_kron`:

The most versatile and easy-to-use PSGD optimizer is `pro`, which uses Procrustes-based 
preconditioners. It has less hyperparameters that need tuning than adam, and can generally act as a 
drop-in replacement.

Distributed kron implements the PRO optimizer meant for large scale distributed training in JAX. It uses blocked
preconditioners, vmapping of layers, partitioning of grads, and sharding constraints to allow for easy and efficient
second-order training of large models.


## Installation

```bash
pip install distributed-kron
```

## Basic Usage

**FYI**: PRO updates the preconditioner every step, providing consistent performance throughout training.

**Learning Rate**: PRO usually works well with learning rates similar to Adam's (e.g., 0.001).

**Weight Decay**: PRO usually likes a weight decay around 0.1 (can be larger than adam's).

For basic usage, use `distributed_kron` like any other optax optimizer:

```python
from distributed_kron import pro

optimizer = pro()
opt_state = optimizer.init(params)

updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

## Distributed Training

See the `kron_example.py` file for a simple example.

The main thing to note is that your workflow should include passing params partition specs into pro through
`params_partition_specs`, which will be used for internal sharding constraints. You can also specify the
`pipeline_axis_name` for pipeline parallelism (typically 'fsdp') and `pipeline_axis_size` for sharding the
preconditioner state across devices.

#### `get_opt_state_partition_specs`:

This is a helper function to get the optimizer state partition specs from the params.

```python
from distributed_kron import get_opt_state_partition_specs

pro_kwargs = dict(
    learning_rate=0.001,
    weight_decay=0.1,
    scanned_layers=scanned_layers_pytree,
    params_partition_specs=params_partition_specs,
    pipeline_axis_name="fsdp",
    pipeline_axis_size=8,
)

optimizer = pro(**pro_kwargs)

opt_state_partition_specs = get_opt_state_partition_specs(
    params=train_state_shapes["params"], **pro_kwargs  # pass in kwargs
)
```

## Hyperparameter Descriptions

`learning_rate`: PRO usually works well with learning rates similar to Adam's (e.g., 0.001).

`weight_decay`: PRO typically likes a weight decay around 0.1, which can be larger than adam's.

`b1`: Momentum coefficient for EMA of gradients (default 0.95).

PRO does not have epsilon or beta2.

**Preconditioner Info:**

*Preconditioner structure*: PRO uses blocked Procrustes-based preconditioners. For a layer with shape (256, 128),
preconditioners are organized into blocks of size `block_size` (default 256). Dimensions larger than `max_size_dense`
(default 16384) automatically use diagonal preconditioners for memory efficiency.

`max_size_dense`: Any dimension with size above this value will have a diagonal preconditioner instead of 
a dense/blocked one. Default is 16384.

`block_size`: Size of blocks for the blocked preconditioner. Default is 256. Larger blocks can be more accurate 
but use more memory.

`preconditioner_lr`: Learning rate for preconditioner updates (default 0.5).

`preconditioner_init_scale`: Initial scale for preconditioner (default 1.0).

`preconditioner_update_style`: Either "PRO" (default) or "QUAD" for the update algorithm.

**Preconditioner updates:**

PRO updates preconditioners every step by default, providing consistent performance throughout training without
needing scheduling.

<hr style="visibility: hidden; margin: 1em 0;">

**Sharding:**

If you are sharding your params, pass your params' `PartitionSpec`s into `pro` through the 
`params_partition_specs` hyperparameter. This will be used for internal sharding constraints.

To shard preconditioners across pipeline stages, use the `pipeline_axis_name` (typically 'fsdp') and 
`pipeline_axis_size` parameters. The preconditioner state will be automatically sharded along the specified axis.

**Scanned layers:**

If you are scanning layers in your network, PRO can also scan over those arrays internally. 
Pass in a pytree the same structure as your params with True values indicating scanned arrays 
and False values indicating non-scanned arrays through the `scanned_layers` hyperparameter. 
PRO will vmap over the first dims of those layers. You can also pass a callable that takes params
and returns such a pytree.

<hr style="visibility: hidden; margin: 1em 0;">

***For more hyperparameter info, please see pro's docstring.***

## Resources

PSGD papers and resources listed from Xi-Lin's repo

1) Xi-Lin Li. Preconditioned stochastic gradient descent,
[arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting
losses and Kronecker product preconditioners.)

2) Xi-Lin Li. Preconditioner on matrix Lie group for SGD,
[arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)

3) Xi-Lin Li. Black box Lie group preconditioners for SGD,
[arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. See
[these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view)
for detailed math derivations.)

4) Xi-Lin Li. Stochastic Hessian fittings on Lie groups,
[arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Some theoretical works on the efficiency of PSGD.
The Hessian fitting problem is shown to be strongly convex on set ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)

5) Omead Pooladzandi, Xi-Lin Li. Curvature-informed SGD via general purpose Lie-group preconditioners,
[arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD
vs. other optimizers.)


## License

[![CC BY 4.0][cc-by-image]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

2024 Evan Walters, Omead Pooladzandi, Xi-Lin Li


[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
