import os
from functools import partial
from pprint import pprint

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
from jax.experimental.mesh_utils import create_device_mesh
import optax

from distributed_kron import kron


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


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
    params_partition_specs = jax.tree.map(lambda x: x.spec, params_sharding)  # only specs, not sharding
    scanned_layers = {"w1_scan": True, "w2": False, "b1": False}  # which arrays in model are scanned
    preconditioner_partition_spec = P("fsdp", None)  # best to explicitly set preconditioner sharding

    kron_kwargs = dict(
        learning_rate=0.0003,
        b1=0.9,
        weight_decay=0.01,
        weight_decay_mask=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        mu_dtype="bfloat16",
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
        params_partition_specs=params_partition_specs,
        preconditioner_partition_spec=preconditioner_partition_spec,
    )

    optimizer = kron(**kron_kwargs)


    @jax.jit
    def init_train_state():
        params = {
            "w1_scan": jnp.ones((2, 512, 1024)),
            "w2": jnp.ones((1024, 500, 2)),
            "b1": jnp.ones(1024),
        }
        # shard params
        params = jax.lax.with_sharding_constraint(params, params_sharding)

        # optimizer state is sharded inside optimizer's init function
        opt_state = optimizer.init(params)

        return {"params": params, "opt_state": opt_state}

    # create train state
    with mesh:
        train_state = init_train_state()
        print("INPUT TRAIN STATE SHAPES:")
        pprint_tree(train_state)

    # grab train state sharding
    train_state_sharding = jax.tree.map(lambda x: x.sharding, train_state)

    @partial(
        jax.jit,
        in_shardings=(params_sharding, train_state_sharding),
        out_shardings=(params_sharding, train_state_sharding),
        donate_argnums=(0, 1),
    )
    def test_step(grads, train_state):
        updates, new_opt_state = optimizer.update(
            grads, train_state["opt_state"], train_state["params"]
        )
        new_params = optax.apply_updates(train_state["params"], updates)

        new_state = {"params": new_params, "opt_state": new_opt_state}

        return updates, new_state


    with mesh:
        grads = jax.tree.map(jnp.ones_like, train_state["params"])
        grads = jax.device_put(grads, device=params_sharding)

        updates, new_state = test_step(grads, train_state)

        """
        In the printout, you will see the preconditioners at
        opt_state.Qs_preconditioners.w1_scan will have a partition spec of
        P('pipeline', None, 'fsdp', None). These dimensions correspond to the
        scanned dim from scanned_layers (0), stacked grad partitions (1), and
        the preconditioner matrix dimensions (2, 3).
        """
        print("OUTPUT UPDATES SHARDING:")
        pprint_tree(updates, shardings=True)
        print("OUTPUT TRAIN STATE SHARDING:")
        pprint_tree(new_state, shardings=True)


def pprint_tree(tree, shardings=False):
    pprint(jax.tree.map(
        lambda x: x.sharding.spec if shardings else x.shape, tree
    ), width=120, sort_dicts=False)


if __name__ == "__main__":
    main(merge_small_dims=True, partition_grads_into_blocks=True)
