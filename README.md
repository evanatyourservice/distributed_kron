# Distributed PSGD Kron

For original PSGD repo and some great resources, see [psgd_torch](https://github.com/lixilinx/psgd_torch).

Implementation of [PSGD Kron](https://github.com/lixilinx/psgd_torch) in JAX (optax-style) for 
distributed training. PSGD is a second-order optimizer originally created by Xi-Lin Li that uses either 
a hessian-based or whitening-based (gg^T) preconditioner, lie groups, and online preconditioner updating 
to improve training convergence, generalization, and efficiency. I highly suggest taking a look at 
Xi-Lin's PSGD repo's readme linked above for interesting details on how PSGD works and experiments using PSGD. 
There are also paper resources listed near the bottom of this readme.

### `distributed_kron`:

The most versatile and easy-to-use PSGD optimizer is `kron`, which uses a Kronecker-factored 
preconditioner. It has less hyperparameters that need tuning than adam, and can generally act as a 
drop-in replacement.

Distributed kron is a version of kron meant for large scale distributed training in JAX. It uses 
merging of dimensions, vmapping of layers, partitioning of grads, and sharding constraints to allow
for efficient, second-order training of large models.


## Installation

```bash
pip install distributed-kron
```

## Basic Usage

FYI: Kron schedules the preconditioner update probability by default to start at 1.0 and anneal to 0.03 
during the first 4k steps of training, so training will be slightly slower at the start but will speed up 
by around 4k steps.

LR: Kron usually likes a learning rate around 3x smaller than adam's. Kron does not share adam's implicit warmup, 
so a longer warmup schedule may be beneficial if divergence is seen early in training.

For basic usage, use `distributed_kron.kron` optimizer like any other optax optimizer:

```python
from distributed_kron import kron

optimizer = kron()
opt_state = optimizer.init(params)

updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

## Advanced Usage

For advanced usage, see the `kron_example.py` file.

The main thing to note is that your workflow should include passing params partition specs into kron through
`params_partition_specs`, which will be used for internal sharding constraints. Also, it is best to explicitly
set preconditioner partition specs using `preconditioner_partition_spec` (see hyperparameters section below).

#### `get_opt_state_partition_specs`:

If you need it, there is an optimizer state partition specs helper function
`get_opt_state_partition_specs`:

```python
from distributed_kron import get_opt_state_partition_specs

kron_kwargs = dict(
    learning_rate=0.0003,
    weight_decay=0.01,
    scanned_layers=scanned_layers_pytree,
    params_partition_specs=params_partition_specs,
    preconditioner_partition_spec=P("fsdp", None),
)

optimizer = kron(**kron_kwargs)

opt_state_partition_specs = get_opt_state_partition_specs(
    params=train_state_shapes["params"], scale_by_kron_only=False, **kron_kwargs  # pass in kwargs
)
```

## Hyperparameters

`learning_rate`: Kron usually likes a learning rate around 3x smaller than adam's. Kron does not share
adam's implicit warmup, so a longer warmup schedule may be beneficial if divergence is seen early in training.

**Preconditioner settings:**

*Preconditioner structure*: For a layer with shape (256, 128), default triangular preconditioners would be shapes
(256, 256) and (128, 128). However, with the following options we can also choose to make some of all of these
preconditioners diagonal, which would be shapes (256,) and (128,).

Depending on how the following settings are chosen, `kron` can balance between memory/speed and effectiveness.
Defaults lead to most precoditioners being triangular except for 1-dimensional layers and very large dimensions.

`max_size_triangular`: Any dimension with size above this value will have a diagonal preconditioner.

`min_ndim_triangular`: Any tensor with less than this number of dimensions will have all diagonal 
preconditioners. Default is 2, so single-dim layers like bias and scale use diagonal preconditioners.

`memory_save_mode`: Can be None, 'one_diag', or 'all_diag'. None is default and lets all 
preconditioners be triangular. 'one_diag' sets the largest or last dim per layer as diagonal 
using `np.argsort(shape)[::-1][0]`. 'all_diag' sets all preconditioners to be diagonal.

**Preconditioner update frequency:**

PSGD generally benefits from more preconditioner updates at the start of training, but once the preconditioner
is learned it's okay to do them less often.

`preconditioner_update_probability`: Kron schedules preconditioner update probability by default using a schedule
that works well for most cases. It anneals from 1 to 0.03 at the beginning of training, so training 
will be slightly slower at the start but will speed up by around 4k steps.

An easy way to adjust update frequency is to pass in your own 
`precond_update_prob_schedule` function to kron's `preconditioner_update_probability` hyperparameter:

```python
from distributed_kron import kron, precond_update_prob_schedule

optimizer = kron(
    preconditioner_update_probability=precond_update_prob_schedule(
        # update precond every 20 steps (default is 0.03)
        min_prob=0.05,
        # update precond every step for 1000 steps before starting to anneal (default is 500)
        flat_start=1000
    )
)
```

This is the default schedule defined in the `precond_update_prob_schedule`:

<img src="assets/default_schedule.png" alt="Default Schedule" width="800" style="max-width: 100%; height: auto;" />

<hr style="visibility: hidden; margin: 1em 0;">

**Sharding:**

If you are sharding your params, pass your params' `PartitionSpec`s into `kron` through the 
`params_partition_specs` hyperparameter. This will be used for internal sharding constraints.

To shard preconditioners, pass a `PartitionSpec` into the `preconditioner_partition_spec` hyperparameter. Best 
practice is to set this to something like `P('fsdp', None)` or `P('fsdp', 'tp')`. If `params_partition_specs`
is set but `preconditioner_partition_spec` is None, a so-so preconditioner sharding strategy will be inferred from 
`params_partition_specs`.

**Scanned layers:**

If you are scanning layers in your network, kron can also scan over those arrays internally. 
Pass in a pytree the same structure as your params with True values indicating scanned arrays 
and False values indicating non-scanned arrays through the `scanned_layers` hyperparameter. 
PSGD will vmap over the first dims of those layers. If you need a more advanced scanning setup, 
please open an issue.

For very large models, the preconditioner update may use too much memory all at once when scanning, 
in which case you can set `lax_map_scanned_layers` to `True` and set `lax_map_batch_size` to a 
reasonable batch size for your setup (`lax.map` scans over batches of vmap, see JAX docs). If 
your net is 32 layers and you're hitting OOM during the optimizer step, you can break the model into
2 or 4 and set `lax_map_batch_size` to 16 or 8 respectively.

<hr style="visibility: hidden; margin: 1em 0;">

***For more hyperparameter info, please see kron's docstring.***

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
