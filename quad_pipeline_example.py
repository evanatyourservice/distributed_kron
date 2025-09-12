import os
from functools import partial
from pprint import pprint

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
from jax.experimental.mesh_utils import create_device_mesh
import optax

from distributed_kron import quad_pipeline_simple as quad
from distributed_kron import get_opt_state_partition_specs_quad_pipeline_simple as get_state_specs


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


def _get(d, k, default=None):
    if not isinstance(d, dict):
        return getattr(d, k, default)
    return d.get(k, default)


def assert_tree_same_structure(a, b):
    ta = jax.tree.structure(a)
    tb = jax.tree.structure(b)
    assert ta == tb, f"Tree structure mismatch:\nactual={ta}\nexpected={tb}"


def pprint_tree(tree, shardings=False):
    def _val(x):
        if shardings:
            try:
                return x.sharding.spec
            except Exception:
                return None
        else:
            try:
                return x.shape
            except Exception:
                return None

    pprint(jax.tree.map(_val, tree), width=120, sort_dicts=False)


def _normalize_spec_for_comparison(spec, axis_name: str, axis_size: int):
    if spec is None:
        return P()

    try:
        parts = list(spec)
    except TypeError:
        return spec

    if axis_size == 1:
        parts = [d for d in parts if d != axis_name]

    while parts and parts[-1] is None:
        parts.pop()

    return P(*parts)


def assert_sharding_matches(tree, expected_spec_tree, axis_name: str, axis_size: int):
    def _leaf_ok(x, s):
        try:
            xs = x.sharding.spec
        except Exception:
            xs = None

        xs_norm = _normalize_spec_for_comparison(xs, axis_name, axis_size)
        s_norm = _normalize_spec_for_comparison(s, axis_name, axis_size)

        assert xs_norm == s_norm, f"Spec mismatch: got {xs}, expected {s} (normalized: {xs_norm} vs {s_norm})"
        return True

    jax.tree.map(_leaf_ok, tree, expected_spec_tree)


def assert_specs_equal(actual_spec_tree, expected_spec_tree, axis_name, axis_size):
    norm = partial(_normalize_spec_for_comparison, axis_name=axis_name, axis_size=axis_size)
    actual_norm = jax.tree.map(norm, actual_spec_tree)
    expected_norm = jax.tree.map(norm, expected_spec_tree)

    assert_tree_same_structure(actual_norm, expected_norm)

    def _assert_equal(a, b):
        assert a == b

    jax.tree.map(_assert_equal, actual_norm, expected_norm)


def build_params_and_sharding(mesh, pipeline_axis_name):
    params = {
        "w1_scan": jnp.ones((2, 100, 128)),
        "w2": jnp.ones((128, 128, 2)),
        "b1": jnp.ones(128),
        "w3": jnp.ones((256, 9000)),
        "s0": jnp.ones(()),
        "v1_scan": jnp.arange(2 * 64, dtype=jnp.float32).reshape(2, 64),
    }

    params_sharding = {
        "w1_scan": NS(mesh, P(None, None, pipeline_axis_name)),
        "w2": NS(mesh, P(None, pipeline_axis_name)),
        "b1": NS(mesh, P(None)),
        "w3": NS(mesh, P(pipeline_axis_name)),
        "s0": NS(mesh, P()),
        "v1_scan": NS(mesh, P(None, pipeline_axis_name)),
    }

    scanned_layers = {"w1_scan": True, "w2": False, "b1": False, "w3": False, "s0": False, "v1_scan": True}

    return params, params_sharding, scanned_layers


class TestScenario:
    def __init__(self, pipeline_axis_size, dtype):
        self.pipeline_axis_size = pipeline_axis_size
        self.dtype = dtype
        self.pipeline_axis_name = "fsdp"

        all_devs = jax.devices()
        if len(all_devs) < self.pipeline_axis_size:
            raise RuntimeError(f"Not enough devices for pipeline_axis_size={self.pipeline_axis_size}; have {len(all_devs)}")
        mesh_devices = create_device_mesh((self.pipeline_axis_size,), devices=all_devs[: self.pipeline_axis_size])
        self.mesh = Mesh(mesh_devices, (self.pipeline_axis_name,))

        self.params, self.params_sharding, scanned_layers = build_params_and_sharding(self.mesh, self.pipeline_axis_name)
        params_partition_specs = jax.tree.map(lambda x: x.spec, self.params_sharding)

        self.quad_kwargs = dict(
            learning_rate=0.0003,
            b1=0.9,
            weight_decay=0.01,
            weight_decay_mask=None,
            max_size_dense=8192,
            dtype=self.dtype,
            scanned_layers=scanned_layers,
            block_size=64,
            params_partition_specs=params_partition_specs,
            pipeline_axis_name=self.pipeline_axis_name,
            pipeline_axis_size=self.pipeline_axis_size,
        )
        self.optimizer = quad(**self.quad_kwargs)

    def run(self):
        print(f"\n=== Scenario: dtype={self.dtype}, pipeline_axis_size={self.pipeline_axis_size} ===")
        with self.mesh:
            self.train_state = self._init_and_validate_state()
            self._step_and_validate_updates()
        print("Scenario PASS\n")

    def _init_and_validate_state(self):
        @jax.jit
        def _init_train_state():
            p = jax.lax.with_sharding_constraint(self.params, self.params_sharding)
            opt_state = self.optimizer.init(p)
            return {"params": p, "opt_state": opt_state}

        train_state = _init_train_state()

        print("INPUT PARAM SHARDING:")
        pprint_tree(jax.tree.map(lambda x: x.sharding, train_state["params"]))

        expected_state_specs = get_state_specs(train_state["params"], **self.quad_kwargs)[0]
        actual_state_sharding = jax.tree.map(lambda x: getattr(x, "sharding", None), train_state["opt_state"][0])
        actual_state_specs = jax.tree.map(lambda ns: getattr(ns, "spec", None), actual_state_sharding)

        assert_specs_equal(actual_state_specs, expected_state_specs, self.pipeline_axis_name, self.pipeline_axis_size)
        print("OPT STATE SPEC FUNCTION MATCH: True")

        return train_state

    def _step_and_validate_updates(self):
        train_state_sharding = jax.tree.map(lambda x: x.sharding, self.train_state)

        @partial(
            jax.jit,
            in_shardings=(self.params_sharding, train_state_sharding),
            out_shardings=(self.params_sharding, train_state_sharding),
            donate_argnums=(0, 1),
        )
        def _test_step(grads, train_state):
            updates, new_opt_state = self.optimizer.update(grads, train_state["opt_state"], train_state["params"])
            new_params = optax.apply_updates(train_state["params"], updates)
            return updates, {"params": new_params, "opt_state": new_opt_state}

        grads = jax.tree.map(jnp.ones_like, self.train_state["params"])
        grads = jax.device_put(grads, device=self.params_sharding)

        updates, new_state = _test_step(grads, self.train_state)

        # --- Validation ---
        assert_tree_same_structure(updates, self.train_state["params"])
        assert_tree_same_structure(new_state["params"], self.train_state["params"])

        any_nonzero = jax.tree_util.tree_reduce(
            lambda acc, x: acc or (jnp.any(x != 0) if hasattr(x, "dtype") else acc), updates, False
        )
        assert bool(any_nonzero), "All updates are zero"
        print("ANY NONZERO UPDATES:", bool(any_nonzero))

        expected_specs = jax.tree.map(lambda ns: ns.spec, self.params_sharding)
        assert_sharding_matches(updates, expected_specs, self.pipeline_axis_name, self.pipeline_axis_size)
        assert_sharding_matches(new_state["params"], expected_specs, self.pipeline_axis_name, self.pipeline_axis_size)

        self._validate_opt_state_sharding(new_state["opt_state"])

        # --- Logging ---
        print("OUTPUT UPDATES SHARDING:")
        pprint_tree(updates, shardings=True)
        print("OUTPUT TRAIN STATE SHARDING:")
        pprint_tree(new_state, shardings=True)

    def _validate_opt_state_sharding(self, opt_state):
        if self.pipeline_axis_size <= 1:
            return

        st = opt_state[0]
        dense_st = st.get("dense", None)
        if dense_st is not None:
            for arr_name in ("Ql", "Qr", "Ll", "Lr"):
                arr = _get(dense_st, arr_name)
                spec = getattr(arr, "sharding", None).spec
                assert spec == P(
                    self.pipeline_axis_name
                ), f"dense concat array '{arr_name}' should be sharded over {self.pipeline_axis_name}, got {spec}"

        for leaf_st in st["large"]:
            if _get(leaf_st, "kind") != "large":
                continue
            for name in ("Ql", "Qr", "Ll", "Lr"):
                arr = _get(leaf_st, name, None)
                if arr is None:
                    continue
                spec = getattr(arr, "sharding", None).spec
                assert spec == P(
                    self.pipeline_axis_name
                ), f"large[{name}] should be sharded over {self.pipeline_axis_name}, got {spec}"


def main():
    scenarios = [(jnp.bfloat16, 1), (jnp.bfloat16, 2), (jnp.float32, 1), (jnp.float32, 2)]
    for dtype, psize in scenarios:
        TestScenario(pipeline_axis_size=psize, dtype=dtype).run()


if __name__ == "__main__":
    main()
