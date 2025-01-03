import os
from time import time
from functools import partial
from pprint import pprint

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
from jax.experimental.mesh_utils import create_device_mesh
import optax

from kron import kron, get_opt_state_partition_specs


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def pprint_tree(tree, shardings=False):
    pprint(jax.tree.map(
        lambda x: x.sharding.spec if shardings else x.shape, tree
    ), width=120, sort_dicts=False)


def main(
    merge_small_dims: bool = True,
    partition_grads_into_blocks: bool = True,
):
    devices = create_device_mesh((2, 2))
    mesh = Mesh(devices, ("fsdp", "pipeline"))

    params_sharding = {
        "w1_scan": NS(mesh, P("pipeline", None, "fsdp")),  # kron maintains pipeline sharding
        "w2": NS(mesh, P("fsdp", None, None)),
        "b1": NS(mesh, P(None)),
    }

    # some inputs for kron
    params_sharding_in = jax.tree.map(lambda x: x.spec, params_sharding)  # only specs, not sharding
    scanned_layers = {"w1_scan": True, "w2": False, "b1": False}  # which layers in model are scanned
    preconditioner_sharding = P("fsdp", None)  # explicitly set sharding for preconditioners

    opt_kwargs = dict(
        learning_rate=0.0003,
        b1=0.9,
        weight_decay=0.01,
        weight_decay_mask=None,
        normalize_grads=True,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        mu_dtype=None,
        precond_dtype=None,
        precond_update_precision="tensorfloat32",
        precond_grads_precision=None,
        scanned_layers=scanned_layers,
        lax_map_scanned_layers=False,
        lax_map_batch_size=8,
        merge_small_dims=merge_small_dims,
        target_merged_dim_size=4096,
        partition_grads_into_blocks=partition_grads_into_blocks,
        block_size=512,
        params_sharding=params_sharding_in,
        preconditioner_sharding=preconditioner_sharding,
    )

    optimizer = kron(**opt_kwargs)

    def init_train_state():
        params = {"w1_scan": jnp.ones((2, 512, 1024)), "w2": jnp.ones((1024, 500, 2)), "b1": jnp.ones(1024)}
        opt_state = optimizer.init(params)
        return {"params": params, "opt_state": opt_state}

    with mesh:
        train_state_shapes = jax.eval_shape(init_train_state)
    opt_state_sharding = get_opt_state_partition_specs(train_state_shapes["params"], **opt_kwargs)
    opt_state_sharding = jax.tree.map(lambda spec: NS(mesh, spec), opt_state_sharding)
    train_state_sharding = {"params": params_sharding, "opt_state": opt_state_sharding}
    init_train_state_jitted = jax.jit(init_train_state, out_shardings=train_state_sharding)

    @partial(
        jax.jit,
        in_shardings=(params_sharding, train_state_sharding),
        # can comment out because optimizer's internal sharding constraints should hold
        # out_shardings=(params_sharding, train_state_sharding),
    )
    def test_step(grads, train_state):
        params = train_state["params"]
        opt_state = train_state["opt_state"]
        updates, new_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return updates, {"params": params, "opt_state": new_state}

    with mesh:
        train_state = init_train_state_jitted()
        print("Input train state shapes:")
        pprint_tree(train_state)

        grads = jax.tree.map(jnp.ones_like, train_state["params"])
        grads = jax.block_until_ready(jax.device_put(grads, device=params_sharding))

        start = time()
        updates, new_state = jax.block_until_ready(test_step(grads, train_state))
        end = time()
        print(f"TIME taken: {end - start} seconds")

        print("Output updates sharding:")
        pprint_tree(updates, shardings=True)
        print("Output train state sharding:")
        pprint_tree(new_state, shardings=True)


if __name__ == "__main__":
    main(merge_small_dims=False, partition_grads_into_blocks=False)
    main(merge_small_dims=True, partition_grads_into_blocks=False)
    main(merge_small_dims=False, partition_grads_into_blocks=True)
    main(merge_small_dims=True, partition_grads_into_blocks=True)