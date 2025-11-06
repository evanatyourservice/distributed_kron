import os
from functools import partial
from pprint import pprint

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
from jax.experimental.mesh_utils import create_device_mesh
import optax

from distributed_kron import pro


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def main():
    devices = create_device_mesh((2, 2))
    mesh = Mesh(devices, ("fsdp", "pipeline"))


    params_sharding = {
        "w1_scan": NS(mesh, P("pipeline", None, "fsdp")),  # pro maintains pipeline sharding
        "w2": NS(mesh, P("fsdp", None, None)),
        "b1": NS(mesh, P(None)),
    }

    # some inputs for pro
    params_partition_specs = jax.tree.map(lambda x: x.spec, params_sharding)  # only specs, not sharding
    scanned_layers = {"w1_scan": True, "w2": False, "b1": False}  # which arrays in model are scanned

    pro_kwargs = dict(
        learning_rate=0.001,
        b1=0.95,
        weight_decay=0.1,
        weight_decay_mask=None,
        max_size_dense=16384,
        preconditioner_lr=0.5,
        preconditioner_init_scale=1.0,
        preconditioner_update_style="PRO",
        dtype="float32",
        scanned_layers=scanned_layers,
        block_size=256,
        pipeline_axis_name="fsdp",
        pipeline_axis_size=2,
        params_partition_specs=params_partition_specs,
        noise_scale=1e-9,
    )

    optimizer = pro(**pro_kwargs)


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
        In the printout, you will see the preconditioner state sharded along
        the pipeline axis (fsdp). PRO maintains efficient sharding of both
        dense and large preconditioner blocks.
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
    main()
