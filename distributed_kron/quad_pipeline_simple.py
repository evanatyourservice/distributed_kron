# simplified pipeline-sharded psgd-quad in jax/optax.
#
# key ideas:
#   - merge tensor dimensions to 2d before grouping.
#   - groups: dense (both axes <= max_size_dense), large (>=1 axis > max_size_dense), and one_d.
#   - block dense axes only. pad edge blocks with zeros (gradients) or zero-padded identity matrices (q).
#   - dense tensors: concatenate all blocks, shard along the pipeline axis, and keep q/l sharded.
#   - large tensors: block only the dense axis, pad each leaf's block stack to the pipeline axis, and shard per-leaf.
#   - one_d tensors: sign(momentum) update (1d whitening).
#   - matmul-based updates (no einsum).
#   - l is f32; q is bf16 or f32.
#
# author: https://github.com/evanatyourservice

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union, List, Dict

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec

from optax._src import base, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax import tree_utils as otu

try:
    import flax.linen as nn
    from flax import struct
    HAVE_FLAX = True
except Exception:
    nn = None  # type: ignore
    HAVE_FLAX = False


# --- state enums (avoid strings in jitted outputs) ---
KIND_ONE_D = 0
KIND_DENSE_META = 1
KIND_LARGE = 2


if HAVE_FLAX:

    @struct.dataclass
    class LeafState:
        # static metadata, not traced
        kind: int = struct.field(pytree_node=False)
        scanned: int = struct.field(pytree_node=False)
        B: int = struct.field(pytree_node=False)
        shape: Optional[Tuple[int, ...]] = struct.field(pytree_node=False, default=None)
        merged: Optional[Tuple[int, ...]] = struct.field(pytree_node=False, default=None)
        nr: Optional[int] = struct.field(pytree_node=False, default=None)
        nc: Optional[int] = struct.field(pytree_node=False, default=None)
        block_size: Optional[int] = struct.field(pytree_node=False, default=None)
        diag_left: Optional[bool] = struct.field(pytree_node=False, default=None)
        diag_right: Optional[bool] = struct.field(pytree_node=False, default=None)
        stack: Optional[int] = struct.field(pytree_node=False, default=None)
        # dynamic tensors, traced
        Ql: Optional[jax.Array] = None
        Qr: Optional[jax.Array] = None
        Ll: Optional[jax.Array] = None
        Lr: Optional[jax.Array] = None

else:
    LeafState = dict


if HAVE_FLAX:

    @struct.dataclass
    class DenseState:
        # dynamic tensors
        Ql: jax.Array
        Qr: jax.Array
        Ll: jax.Array
        Lr: jax.Array
        valid_rows: jax.Array
        valid_cols: jax.Array
        # static metadata
        valid_count: int = struct.field(pytree_node=False)
        block_size: int = struct.field(pytree_node=False)


# --- main helpers ---


def _get(d, k, default=None):
    if not isinstance(d, dict):
        return getattr(d, k, default)
    return d.get(k, default)


def _merge_dims(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """merge tensor dims into at most 2d."""
    if len(shape) < 2:
        return shape
    if int(np.prod(shape)) == int(np.max(shape)):
        # 1d-like: at most one non-1 dimension, so treat as 1d
        return (int(np.max(shape)),)
    if len(shape) == 2:
        return tuple(map(int, shape))
    dims = list(map(int, shape))
    best_ratio, best_split = float("inf"), 1
    for s in range(1, len(dims)):
        lp, rp = int(np.prod(dims[:s])), int(np.prod(dims[s:]))
        ratio = max(lp, rp) / min(lp, rp)
        if ratio < best_ratio:
            best_ratio, best_split = ratio, s
    return (int(np.prod(dims[:best_split])), int(np.prod(dims[best_split:])))


def _shard(x, spec_tree: Optional[Union[PartitionSpec, List, Tuple, Dict]]):
    if spec_tree is None:
        return x

    def _apply(value, spec):
        if spec is None:
            return value
        try:
            return with_sharding_constraint(value, spec)
        except RuntimeError as e:
            if "requires a non-empty mesh" in str(e):
                return value
            raise

    if isinstance(spec_tree, PartitionSpec):
        leaves, tdef = jax.tree.flatten(x)
        return tdef.unflatten([_apply(v, spec_tree) for v in leaves])

    try:
        x_leaves, x_def = jax.tree.flatten(x)
        spec_leaves, spec_def = jax.tree.flatten(spec_tree)
        if x_def == spec_def:
            return x_def.unflatten([_apply(v, s) for v, s in zip(x_leaves, spec_leaves)])
    except Exception as e:
        if isinstance(e, RuntimeError) and "requires a non-empty mesh" in str(e):
            return x
        print(e)
        print(f"t item types: {[type(item) for item in x_leaves]}")
        try:
            print(f"s item types: {[type(item) for item in spec_leaves]}")
        except Exception:
            pass
        raise ValueError(
            "sharding spec pytree mismatch: pass a single partitionspec or an aligned pytree (allow none to skip).\n"
            f"x tree def: {x_def}\n"
            f"spec_tree tree def: {spec_def}"
        )


def _identity_padded(block_size: int, valid: int, dtype: jnp.dtype) -> jax.Array:
    """identity of size `valid` padded to (block_size, block_size) with zeros."""
    if valid >= block_size:
        return jnp.eye(block_size, dtype=dtype)
    eye = jnp.eye(valid, dtype=dtype)
    return jnp.pad(eye, ((0, block_size - valid), (0, block_size - valid)), constant_values=0)


# --- blocking helpers (dense axes only) ---


def _block2d_full(x: jax.Array, block_size: int) -> Tuple[jax.Array, Tuple[int, int, int, int]]:
    """block both axes to block_size. returns:
    blocks: [nr*nc, bs, bs], meta=(nr, nc, m, n) with original m,n."""
    m, n = int(x.shape[-2]), int(x.shape[-1])
    nr, nc = (m + block_size - 1) // block_size, (n + block_size - 1) // block_size
    pm, pn = nr * block_size, nc * block_size
    dm, dn = pm - m, pn - n
    xpad = jnp.pad(x, ((0, dm), (0, dn)))
    # reshape to (nr, bs, nc, bs) -> (nr, nc, bs, bs) -> (nr*nc, bs, bs)
    x4 = xpad.reshape(nr, block_size, nc, block_size).transpose(0, 2, 1, 3)
    return x4.reshape(nr * nc, block_size, block_size), (nr, nc, m, n)


def _unblock2d_full(blocks: jax.Array, meta: Tuple[int, int, int, int], block_size: int) -> jax.Array:
    """inverse of _block2d_full. returns [m, n] cropped to original."""
    nr, nc, m, n = meta
    x4 = blocks.reshape(nr, nc, block_size, block_size).transpose(0, 2, 1, 3)
    x = x4.reshape(nr * block_size, nc * block_size)
    return x[:m, :n]


def _block_cols(x: jax.Array, block_size: int) -> Tuple[jax.Array, Tuple[int, int, int]]:
    """split columns only. returns blocks: [nc, m, bs], meta=(nc, m, n)."""
    m, n = int(x.shape[-2]), int(x.shape[-1])
    nc = (n + block_size - 1) // block_size
    pn = nc * block_size
    dn = pn - n
    xpad = jnp.pad(x, ((0, 0), (0, dn)))
    # reshape columns into (nc, bs)
    xp = xpad.reshape(m, nc, block_size).transpose(1, 0, 2)
    return xp, (nc, m, n)


def _unblock_cols(blocks: jax.Array, meta: Tuple[int, int, int], block_size: int) -> jax.Array:
    """inverse of _block_cols. returns [m, n] cropped to original."""
    nc, m, n = meta
    x = blocks.transpose(1, 0, 2).reshape(m, nc * block_size)
    return x[:, :n]


def _block_rows(x: jax.Array, block_size: int) -> Tuple[jax.Array, Tuple[int, int, int]]:
    """split rows only. returns blocks: [nr, bs, n], meta=(nr, m, n)."""
    m, n = int(x.shape[-2]), int(x.shape[-1])
    nr = (m + block_size - 1) // block_size
    pm = nr * block_size
    dm = pm - m
    xpad = jnp.pad(x, ((0, dm), (0, 0)))
    # reshape rows into (nr, bs)
    xp = xpad.reshape(nr, block_size, n)
    return xp, (nr, m, n)


def _unblock_rows(blocks: jax.Array, meta: Tuple[int, int, int], block_size: int) -> jax.Array:
    """inverse of _block_rows. returns [m, n] cropped to original."""
    nr, m, n = meta
    x = blocks.reshape(nr * block_size, n)
    return x[:m, :n]


# --- quad updates ---


def _diag_update(term1: jax.Array, term2: jax.Array, L: jax.Array, Q: jax.Array, lr_precond: jax.Array):
    ell = jnp.max(term1) + term2
    Ln = jnp.maximum(0.95 * L + 0.05 * ell, ell)
    z = (lr_precond / (2.0 * Ln)).astype(Q.dtype)
    gain = 1.0 - z * (term1 - term2)
    Qn = Q * (gain * gain)
    return Qn, Ln


def _lb_spectral_outer(A: jax.Array) -> jax.Array:
    """lower bound on spectral norm."""
    scale = jnp.max(jnp.diag(A))
    A /= scale
    j = jnp.argmax(jnp.sum(A * A, axis=1))
    x = A @ jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
    x = x / jnp.linalg.norm(x)
    return jnp.linalg.norm(A @ x) * scale


def _dense_update(term_outer: jax.Array, term2: jax.Array, L: jax.Array, Q: jax.Array, lr_precond: jax.Array):
    ell = _lb_spectral_outer(term_outer) + term2
    Ln = jnp.maximum(0.95 * L + 0.05 * ell, ell)
    z = (lr_precond / (2.0 * Ln)).astype(Q.dtype)
    P = Q - z * (term_outer @ Q - term2 * Q)
    P = P - z * (P @ term_outer - P * term2)
    Qn = (P + P.T) / 2.0
    return Qn, Ln


def _precondition_2d_one(
    key: jax.Array,
    Ql: jax.Array,
    Qr: jax.Array,
    Ll: jax.Array,
    Lr: jax.Array,
    G: jax.Array,
    valid_shape: Tuple[int, int],
    diag_left: bool,
    diag_right: bool,
    lr_precond: jax.Array,
    noise_scale: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """single-sample 2d preconditioning and q/l updates."""
    m, n = valid_shape[0], valid_shape[1]
    noise = jax.random.normal(key, G.shape, G.dtype) * noise_scale
    rows = jnp.arange(G.shape[0]) < m
    cols = jnp.arange(G.shape[1]) < n
    mask = rows[:, None] & cols[None, :]
    Gn = G + noise * mask
    
    total_numel = m * n

    if not diag_left and not diag_right:
        # DD
        Pg = jax.numpy.linalg.multi_dot([Ql, Ql, Gn, Qr, Qr])
        termL = Pg @ Pg.T
        termR = Pg.T @ Pg
        term2L = total_numel / m
        term2R = total_numel / n
        Ql_new, Ll_new = _dense_update(termL, term2L, Ll, Ql, lr_precond)
        Qr_new, Lr_new = _dense_update(termR, term2R, Lr, Qr, lr_precond)
        Pg_out = jax.numpy.linalg.multi_dot([Ql_new, Ql_new, G, Qr_new, Qr_new])

    elif diag_left and not diag_right:
        # dD
        Pg = (Ql * Ql)[:, None] * jax.numpy.linalg.multi_dot([Gn, Qr, Qr])
        termL = jnp.sum(Pg * Pg, axis=1)
        termR = Pg.T @ Pg
        term2L = total_numel / m
        term2R = total_numel / n
        Ql_new, Ll_new = _diag_update(termL, term2L, Ll, Ql, lr_precond)
        Qr_new, Lr_new = _dense_update(termR, term2R, Lr, Qr, lr_precond)
        Pg_out = (Ql_new * Ql_new)[:, None] * jax.numpy.linalg.multi_dot([G, Qr_new, Qr_new])

    elif not diag_left and diag_right:
        # Dd
        Pg = jax.numpy.linalg.multi_dot([Ql, Ql, Gn]) * (Qr * Qr)[None, :]
        termL = Pg @ Pg.T
        termR = jnp.sum(Pg * Pg, axis=0)
        term2L = total_numel / m
        term2R = total_numel / n
        Ql_new, Ll_new = _dense_update(termL, term2L, Ll, Ql, lr_precond)
        Qr_new, Lr_new = _diag_update(termR, term2R, Lr, Qr, lr_precond)
        Pg_out = jax.numpy.linalg.multi_dot([Ql_new, Ql_new, G]) * (Qr_new * Qr_new)[None, :]

    else:
        # dd
        Pg = (Ql * Ql)[:, None] * Gn * (Qr * Qr)[None, :]
        termL = jnp.sum(Pg * Pg, axis=1)
        termR = jnp.sum(Pg * Pg, axis=0)
        term2L = total_numel / m
        term2R = total_numel / n
        Ql_new, Ll_new = _diag_update(termL, term2L, Ll, Ql, lr_precond)
        Qr_new, Lr_new = _diag_update(termR, term2R, Lr, Qr, lr_precond)
        Pg_out = (Ql_new * Ql_new)[:, None] * G * (Qr_new * Qr_new)[None, :]

    return Ql_new, Qr_new, Ll_new, Lr_new, Pg_out


# --- optimizer: simplified pipeline-sharded psgd-quad ---


def scale_by_quad(
    lr_style: Optional[str] = "adam",
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: Union[str, jnp.dtype] = jnp.bfloat16,
    scanned_layers: Optional[base.Params] = None,
    block_size: int = 256,
    pipeline_axis_name: Optional[str] = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: Optional[Union[PartitionSpec, List, Tuple, Dict]] = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    """main quad preconditioner (simplified)."""

    dtype = canonicalize_dtype(dtype)
    assert dtype in (jnp.bfloat16, jnp.float32), "dtype must be bfloat16 or float32"
    assert block_size is None or block_size > 0, "block_size must be positive"

    # init: build q/l state
    def init_fn(params):
        # optionally unbox flax partitioned
        if HAVE_FLAX:
            params_unboxed = jax.tree.map(
                lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
                params,
                is_leaf=lambda x: isinstance(x, nn.Partitioned),
            )
        else:
            params_unboxed = params

        # momentum buffers (original shapes/specs)
        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=dtype), params_unboxed)
            if params_partition_specs is not None:
                mu = _shard(mu, params_partition_specs)

        # helper flags and shapes
        scanned_flags = scanned_layers if scanned_layers is not None else jax.tree.map(lambda _: False, params_unboxed)

        # build dense concatenated q/l state and large per-leaf q/l state
        dense_Ql_list: List[jax.Array] = []
        dense_Qr_list: List[jax.Array] = []
        dense_Ll_list: List[jax.Array] = []
        dense_Lr_list: List[jax.Array] = []

        dense_valid_rc: List[Tuple[int, int]] = []  # valid rows/cols per block for masking edge blocks

        large_state: List[Any] = []  # per-leaf large state

        # traverse leaves
        leaves, tdef = jax.tree.flatten(params_unboxed)
        flags, _ = jax.tree.flatten(scanned_flags)

        for leaf, scanned in zip(leaves, flags):
            # add dummy [1] batch for non-scanned
            p = leaf if scanned else leaf[None, ...]
            B = int(p.shape[0])
            shape_wo = tuple(map(int, p.shape[1:]))

            # merge dims (no leading batch)
            merged = _merge_dims(shape_wo)
            if len(merged) <= 1:
                # one_d uses sign(momentum) — no q/l state needed
                large_state.append(LeafState(kind=KIND_ONE_D, scanned=int(scanned), B=B, shape=shape_wo))
                continue

            m, n = map(int, merged)
            is_large_m = m > max_size_dense
            is_large_n = n > max_size_dense
            is_dense = (not is_large_m) and (not is_large_n)

            if is_dense:
                # DD
                # for each sample, block both axes to block_size
                nr, nc = (m + block_size - 1) // block_size, (n + block_size - 1) // block_size

                # per-row/col valid sizes for edge blocks
                row_sizes = [block_size] * (nr - 1) + [m - block_size * (nr - 1) if nr > 0 else 0]
                col_sizes = [block_size] * (nc - 1) + [n - block_size * (nc - 1) if nc > 0 else 0]
                row_sizes = [rs if rs > 0 else block_size for rs in row_sizes]
                col_sizes = [cs if cs > 0 else block_size for cs in col_sizes]

                # construct identity-padded q blocks and zero l
                for _b in range(B):
                    for ri in range(nr):
                        for cj in range(nc):
                            vr, vc = row_sizes[ri], col_sizes[cj]
                            Ql = _identity_padded(block_size, vr, dtype)
                            Qr = _identity_padded(block_size, vc, dtype)
                            dense_Ql_list.append(Ql)
                            dense_Qr_list.append(Qr)
                            dense_Ll_list.append(jnp.zeros([], jnp.float32))
                            dense_Lr_list.append(jnp.zeros([], jnp.float32))
                            dense_valid_rc.append((vr, vc))
                large_state.append(
                    LeafState(
                        kind=KIND_DENSE_META, scanned=int(scanned), B=B, merged=(m, n), nr=nr, nc=nc, block_size=block_size
                    )
                )

            else:
                # large: at least one large axis — block the dense axis only
                diag_left = is_large_m
                diag_right = is_large_n

                if diag_left and diag_right:
                    # dd: no blocking
                    Ql = jnp.ones((B, m), dtype=dtype)
                    Qr = jnp.ones((B, n), dtype=dtype)
                    Ll = jnp.zeros((B,), jnp.float32)
                    Lr = jnp.zeros((B,), jnp.float32)
                    large_state.append(
                        LeafState(
                            kind=KIND_LARGE,
                            scanned=int(scanned),
                            B=B,
                            merged=(m, n),
                            diag_left=True,
                            diag_right=True,
                            Ql=Ql,
                            Qr=Qr,
                            Ll=Ll,
                            Lr=Lr,
                            stack=B,
                        )
                    )

                elif diag_left != diag_right:
                    # dD or Dd
                    block_rows = not diag_left
                    dim_to_block = m if block_rows else n
                    other_dim = n if block_rows else m

                    num_blocks_per_sample = (dim_to_block + block_size - 1) // block_size
                    stack = B * num_blocks_per_sample

                    Q_diag = jnp.broadcast_to(jnp.ones((1, other_dim), dtype=dtype), (stack, other_dim))

                    Q_blocked_blocks = []
                    for _ in range(B):
                        for i in range(num_blocks_per_sample):
                            v = (
                                block_size
                                if i < num_blocks_per_sample - 1
                                else (
                                    dim_to_block - block_size * (num_blocks_per_sample - 1)
                                    if num_blocks_per_sample > 0
                                    else block_size
                                )
                            )
                            v = v if v > 0 else block_size
                            Q_blocked_blocks.append(_identity_padded(block_size, v, dtype))
                    Q_blocked = jnp.stack(Q_blocked_blocks, axis=0)

                    Ql = Q_blocked if block_rows else Q_diag
                    Qr = Q_diag if block_rows else Q_blocked
                    Ll = jnp.zeros((stack,), jnp.float32)
                    Lr = jnp.zeros((stack,), jnp.float32)

                    large_state.append(
                        LeafState(
                            kind=KIND_LARGE,
                            scanned=int(scanned),
                            B=B,
                            merged=(m, n),
                            diag_left=diag_left,
                            diag_right=diag_right,
                            Ql=Ql,
                            Qr=Qr,
                            Ll=Ll,
                            Lr=Lr,
                            stack=stack,
                            nr=num_blocks_per_sample if block_rows else None,
                            nc=num_blocks_per_sample if not block_rows else None,
                            block_size=block_size,
                        )
                    )
                else:
                    # should not happen: handled by is_dense above
                    raise AssertionError("unexpected large case.")

        # concatenate dense q/l across all leaves/samples and pad to pipeline axis size
        if dense_Ql_list:
            Ql_cat = jnp.stack(dense_Ql_list, axis=0)
            Qr_cat = jnp.stack(dense_Qr_list, axis=0)
            Ll_cat = jnp.stack(dense_Ll_list, axis=0)
            Lr_cat = jnp.stack(dense_Lr_list, axis=0)

            valid_rows = jnp.array([vr for (vr, _) in dense_valid_rc], dtype=jnp.int32)
            valid_cols = jnp.array([vc for (_, vc) in dense_valid_rc], dtype=jnp.int32)

            valid_count = Ql_cat.shape[0]
            if pipeline_axis_size > 1:
                pad = (-valid_count) % pipeline_axis_size
            else:
                pad = 0
            if pad > 0:
                # fake entries: identity q, ones l, ones grads
                eye = jnp.eye(block_size, dtype=dtype)
                Ql_pad = jnp.broadcast_to(eye, (pad, block_size, block_size))
                Qr_pad = jnp.broadcast_to(eye, (pad, block_size, block_size))
                Ll_pad = jnp.ones((pad,), jnp.float32)
                Lr_pad = jnp.ones((pad,), jnp.float32)
                Ql_cat = jnp.concatenate([Ql_cat, Ql_pad], axis=0)
                Qr_cat = jnp.concatenate([Qr_cat, Qr_pad], axis=0)
                Ll_cat = jnp.concatenate([Ll_cat, Ll_pad], axis=0)
                Lr_cat = jnp.concatenate([Lr_cat, Lr_pad], axis=0)
                valid_rows = jnp.concatenate([valid_rows, jnp.full((pad,), block_size, jnp.int32)], axis=0)
                valid_cols = jnp.concatenate([valid_cols, jnp.full((pad,), block_size, jnp.int32)], axis=0)

            # shard along leading dim
            if pipeline_axis_name is not None:
                Ql_cat = _shard(Ql_cat, PartitionSpec(pipeline_axis_name))
                Qr_cat = _shard(Qr_cat, PartitionSpec(pipeline_axis_name))
                Ll_cat = _shard(Ll_cat, PartitionSpec(pipeline_axis_name))
                Lr_cat = _shard(Lr_cat, PartitionSpec(pipeline_axis_name))
                valid_rows = _shard(valid_rows, PartitionSpec(pipeline_axis_name))
                valid_cols = _shard(valid_cols, PartitionSpec(pipeline_axis_name))
            if HAVE_FLAX:
                dense_state = DenseState(
                    Ql=Ql_cat,
                    Qr=Qr_cat,
                    Ll=Ll_cat,
                    Lr=Lr_cat,
                    valid_rows=valid_rows,
                    valid_cols=valid_cols,
                    valid_count=int(valid_count),  # static
                    block_size=int(block_size),  # static
                )
            else:
                dense_state = dict(
                    Ql=Ql_cat,
                    Qr=Qr_cat,
                    Ll=Ll_cat,
                    Lr=Lr_cat,
                    valid_rows=valid_rows,
                    valid_cols=valid_cols,
                    valid_count=int(valid_count),  # static
                    block_size=int(block_size),  # static
                )
        else:
            dense_state = None

        # pad/shard large leaves per-leaf along their leading "stack" (already computed)
        for i, st in enumerate(large_state):
            if st.kind != KIND_LARGE:
                continue

            updates = {}
            current_Ql, current_Qr, current_Ll, current_Lr = st.Ql, st.Qr, st.Ll, st.Lr
            current_stack = st.stack

            if pipeline_axis_size > 1:
                pad = (-current_stack) % pipeline_axis_size
            else:
                pad = 0

            if pad > 0:
                if st.diag_left and st.diag_right:
                    current_Ql = jnp.pad(st.Ql, ((0, pad), (0, 0)), constant_values=1.0)
                    current_Qr = jnp.pad(st.Qr, ((0, pad), (0, 0)), constant_values=1.0)
                elif st.diag_left and (not st.diag_right):
                    eye = jnp.eye(st.block_size, dtype=dtype)
                    current_Ql = jnp.pad(st.Ql, ((0, pad), (0, 0)), constant_values=1.0)
                    current_Qr = jnp.concatenate([st.Qr, jnp.broadcast_to(eye, (pad, eye.shape[0], eye.shape[1]))], 0)
                elif (not st.diag_left) and st.diag_right:
                    eye = jnp.eye(st.block_size, dtype=dtype)
                    current_Ql = jnp.concatenate([st.Ql, jnp.broadcast_to(eye, (pad, eye.shape[0], eye.shape[1]))], 0)
                    current_Qr = jnp.pad(st.Qr, ((0, pad), (0, 0)), constant_values=1.0)
                else:
                    raise AssertionError
                current_Ll = jnp.pad(st.Ll, ((0, pad),), constant_values=1.0)
                current_Lr = jnp.pad(st.Lr, ((0, pad),), constant_values=1.0)
                current_stack += pad

            updates["Ql"] = current_Ql
            updates["Qr"] = current_Qr
            updates["Ll"] = current_Ll
            updates["Lr"] = current_Lr
            updates["stack"] = current_stack

            if pipeline_axis_name is not None:
                updates["Ql"] = _shard(updates["Ql"], PartitionSpec(pipeline_axis_name))
                updates["Qr"] = _shard(updates["Qr"], PartitionSpec(pipeline_axis_name))
                updates["Ll"] = _shard(updates["Ll"], PartitionSpec(pipeline_axis_name))
                updates["Lr"] = _shard(updates["Lr"], PartitionSpec(pipeline_axis_name))

            large_state[i] = st.replace(**updates)

        opt_state = dict(count=jnp.zeros([], jnp.int32), mu=mu, large=large_state)
        if dense_state is not None:
            opt_state["dense"] = dense_state
        return opt_state

    # update: apply quad preconditioning
    def update_fn(updates: base.Updates, state: dict, params: base.Params | None = None):
        step = safe_int32_increment(state["count"])
        plr = jnp.maximum(preconditioner_lr * jax.lax.rsqrt(1.0 + step / 10000.0), 0.3)

        flax_partitioned = False
        if HAVE_FLAX:
            flat, tdef = jax.tree.flatten(updates, is_leaf=lambda g: isinstance(g, (nn.Partitioned, jax.ShapeDtypeStruct)))
            if any(isinstance(g, nn.Partitioned) for g in flat):
                updates = tdef.unflatten([g.unbox() if isinstance(g, nn.Partitioned) else g for g in flat])
                flax_partitioned = True

        # momentum
        mu = state["mu"]
        mupd = updates
        if mu is not None and b1 > 0:
            mu = otu.tree_update_moment(updates, mu, b1, 1)
            if params_partition_specs is not None:
                mu = _shard(mu, params_partition_specs)
            mupd = otu.tree_bias_correction(mu, b1, step)
        mu = otu.tree_cast(mu, dtype) if mu is not None else None
        mupd = otu.tree_cast(mupd, dtype)

        if normalize_grads:
            mupd = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-6).astype(g.dtype), mupd)

        leaves_u, tdef_u = jax.tree.flatten(mupd)
        perleaf_state: List[Any] = state["large"]

        # dense path: gather, concat, pad, shard, update, scatter
        dense_state = state.get("dense")
        pg_dense_blocks: jax.Array | None = None
        dense_block_count = 0

        if dense_state is not None:
            # gather blocks in the same order as init
            blocks_list = []
            for leaf, st in zip(leaves_u, perleaf_state):
                if st.kind != KIND_DENSE_META:
                    continue  # not dense_meta
                B = st.B
                m, n = st.merged
                nr, nc = st.nr, st.nc
                x2d = jnp.reshape(leaf, (B, m, n))
                current_block_size = int(_get(dense_state, "block_size"))
                for b in range(B):
                    blocks, _ = _block2d_full(x2d[b], current_block_size)
                    blocks_list.append(blocks)
            if blocks_list:
                grads_cat = jnp.concatenate(blocks_list, axis=0)
                dense_block_count = grads_cat.shape[0]
                # pad to state length if needed
                state_len = _get(dense_state, "Ql").shape[0]
                if dense_block_count < state_len:
                    pad = state_len - dense_block_count
                    grads_cat = jnp.concatenate(
                        [grads_cat, jnp.ones((pad, current_block_size, current_block_size), grads_cat.dtype)], axis=0
                    )
                elif dense_block_count > state_len:
                    raise ValueError(
                        "dense concatenation produced more blocks than q state. check block_size/grouping consistency."
                    )
                if pipeline_axis_name is not None:
                    grads_cat = _shard(grads_cat, PartitionSpec(pipeline_axis_name))

                # per-sample keys
                key_dense = jax.random.fold_in(jax.random.PRNGKey(42), step)
                keys = jax.random.split(key_dense, grads_cat.shape[0])
                if pipeline_axis_name is not None:
                    keys = _shard(keys, PartitionSpec(pipeline_axis_name))

                # vmap over blocks
                diag_left = False
                diag_right = False
                valid_shape_dense = jnp.stack([_get(dense_state, "valid_rows"), _get(dense_state, "valid_cols")], axis=1)
                if pipeline_axis_name is not None:
                    # ensure all leading-axis vmapped inputs are constrained
                    valid_shape_dense = _shard(valid_shape_dense, PartitionSpec(pipeline_axis_name))
                    Ql_in = _shard(_get(dense_state, "Ql"), PartitionSpec(pipeline_axis_name))
                    Qr_in = _shard(_get(dense_state, "Qr"), PartitionSpec(pipeline_axis_name))
                    Ll_in = _shard(_get(dense_state, "Ll"), PartitionSpec(pipeline_axis_name))
                    Lr_in = _shard(_get(dense_state, "Lr"), PartitionSpec(pipeline_axis_name))
                else:
                    Ql_in = _get(dense_state, "Ql")
                    Qr_in = _get(dense_state, "Qr")
                    Ll_in = _get(dense_state, "Ll")
                    Lr_in = _get(dense_state, "Lr")
                Ql_new, Qr_new, Ll_new, Lr_new, Pg_cat = vmap(
                    _precondition_2d_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None)
                )(
                    keys,
                    Ql_in,
                    Qr_in,
                    Ll_in,
                    Lr_in,
                    grads_cat,
                    valid_shape_dense,
                    diag_left,
                    diag_right,
                    plr,
                    noise_scale,
                )
                if pipeline_axis_name is not None:
                    Pg_cat = _shard(Pg_cat, PartitionSpec(pipeline_axis_name))


                # write back q/l
                if not isinstance(dense_state, dict):
                    state["dense"] = dense_state.replace(
                        Ql=_shard(otu.tree_cast(Ql_new, dtype), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Ql_new, dtype),
                        Qr=_shard(otu.tree_cast(Qr_new, dtype), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Qr_new, dtype),
                        Ll=_shard(otu.tree_cast(Ll_new, jnp.float32), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Ll_new, jnp.float32),
                        Lr=_shard(otu.tree_cast(Lr_new, jnp.float32), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Lr_new, jnp.float32),
                    )
                else:
                    ps = PartitionSpec(pipeline_axis_name) if pipeline_axis_name is not None else None
                    state["dense"]["Ql"] = _shard(otu.tree_cast(Ql_new, dtype), ps) if ps is not None else otu.tree_cast(Ql_new, dtype)
                    state["dense"]["Qr"] = _shard(otu.tree_cast(Qr_new, dtype), ps) if ps is not None else otu.tree_cast(Qr_new, dtype)
                    state["dense"]["Ll"] = _shard(otu.tree_cast(Ll_new, jnp.float32), ps) if ps is not None else otu.tree_cast(Ll_new, jnp.float32)
                    state["dense"]["Lr"] = _shard(otu.tree_cast(Lr_new, jnp.float32), ps) if ps is not None else otu.tree_cast(Lr_new, jnp.float32)

                # take only valid blocks for scatter
                valid_count = int(_get(dense_state, "valid_count"))
                Pg_cat = Pg_cat[:valid_count]
                pg_dense_blocks = Pg_cat

                # reconstruct per-sample 2d and scatter
                start_idx = 0
                for leaf_idx, (leaf, st) in enumerate(zip(leaves_u, perleaf_state)):
                    if st.kind != KIND_DENSE_META:
                        continue
                    B = st.B
                    m, n = st.merged
                    nr, nc = st.nr, st.nc
                    rec_samples = []
                    for _b in range(B):
                        nb = nr * nc
                        blocks = pg_dense_blocks[start_idx : start_idx + nb]
                        start_idx += nb
                        rec2d = _unblock2d_full(blocks, (nr, nc, m, n), _get(dense_state, "block_size"))
                        rec_samples.append(rec2d)
                    rec = jnp.stack(rec_samples, axis=0)  # [B, m, n]
                    # reshape back to original shape
                    leaves_u[leaf_idx] = jnp.reshape(rec, leaf.shape)

        # large path
        for leaf_idx, (leaf, st) in enumerate(zip(leaves_u, perleaf_state)):
            if st.kind != KIND_LARGE:
                continue  # not a LARGE leaf
            B = st.B
            m, n = st.merged
            diag_left = st.diag_left
            diag_right = st.diag_right
            p2d = jnp.reshape(leaf, (B, m, n))

            if diag_left and diag_right:
                # stack = b, no blocking
                Gs = p2d
                # pad stack already in state
                stack = st.stack
                if Gs.shape[0] < stack:
                    pad = stack - Gs.shape[0]
                    Gs = jnp.concatenate([Gs, jnp.ones((pad, m, n), Gs.dtype)], axis=0)
                if pipeline_axis_name is not None:
                    Gs = _shard(Gs, PartitionSpec(pipeline_axis_name))

                key = jax.random.fold_in(jax.random.PRNGKey(43), step)
                keys = jax.random.split(key, stack)
                if pipeline_axis_name is not None:
                    keys = _shard(keys, PartitionSpec(pipeline_axis_name))

                # ensure vmapped inputs are constrained along leading axis
                if pipeline_axis_name is not None:
                    Ql_in = _shard(st.Ql, PartitionSpec(pipeline_axis_name))
                    Qr_in = _shard(st.Qr, PartitionSpec(pipeline_axis_name))
                    Ll_in = _shard(st.Ll, PartitionSpec(pipeline_axis_name))
                    Lr_in = _shard(st.Lr, PartitionSpec(pipeline_axis_name))
                else:
                    Ql_in, Qr_in, Ll_in, Lr_in = st.Ql, st.Qr, st.Ll, st.Lr

                Ql_new, Qr_new, Ll_new, Lr_new, Pg = vmap(
                    _precondition_2d_one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None)
                )(keys, Ql_in, Qr_in, Ll_in, Lr_in, Gs, (m, n), True, True, plr, noise_scale)
                if pipeline_axis_name is not None:
                    Pg = _shard(Pg, PartitionSpec(pipeline_axis_name))

                state["large"][leaf_idx] = st.replace(
                    Ql=_shard(otu.tree_cast(Ql_new, dtype), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Ql_new, dtype),
                    Qr=_shard(otu.tree_cast(Qr_new, dtype), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Qr_new, dtype),
                    Ll=_shard(otu.tree_cast(Ll_new, jnp.float32), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Ll_new, jnp.float32),
                    Lr=_shard(otu.tree_cast(Lr_new, jnp.float32), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Lr_new, jnp.float32),
                )

                Pg = Pg[:B]
                leaves_u[leaf_idx] = jnp.reshape(Pg, leaf.shape)

            elif diag_left != diag_right:
                # dD or Dd
                block_rows = not diag_left
                block_fn = _block_rows if block_rows else _block_cols
                unblock_fn = _unblock_rows if block_rows else _unblock_cols
                num_blocks_per_sample = st.nr if block_rows else st.nc
                other_dim = n if block_rows else m

                blocks_all = []
                metas = []
                for b in range(B):
                    blocks, meta = block_fn(p2d[b], block_size)
                    blocks_all.append(blocks)
                    metas.append(meta)

                Gs = jnp.concatenate(blocks_all, axis=0)
                stack = st.stack
                if Gs.shape[0] < stack:
                    pad = stack - Gs.shape[0]
                    pad_shape = (pad, block_size, other_dim) if block_rows else (pad, other_dim, block_size)
                    Gs = jnp.concatenate([Gs, jnp.ones(pad_shape, Gs.dtype)], axis=0)

                if pipeline_axis_name is not None:
                    Gs = _shard(Gs, PartitionSpec(pipeline_axis_name))

                key_val = 45 if block_rows else 44
                key = jax.random.fold_in(jax.random.PRNGKey(key_val), step)
                keys = jax.random.split(key, stack)
                if pipeline_axis_name is not None:
                    keys = _shard(keys, PartitionSpec(pipeline_axis_name))

                dim_to_block = m if block_rows else n
                if num_blocks_per_sample > 0:
                    last_block_v = dim_to_block - block_size * (num_blocks_per_sample - 1)
                    v_one_sample = jnp.full((num_blocks_per_sample,), block_size, dtype=jnp.int32).at[-1].set(last_block_v)
                else:
                    v_one_sample = jnp.array([], dtype=jnp.int32)

                v_all_samples = jnp.tile(v_one_sample, B)
                other_dim_arr = jnp.full_like(v_all_samples, other_dim)

                if block_rows:
                    valid_shape_large = jnp.stack([v_all_samples, other_dim_arr], axis=1)
                else:
                    valid_shape_large = jnp.stack([other_dim_arr, v_all_samples], axis=1)

                # Pad to match sharded stack size
                if valid_shape_large.shape[0] < stack:
                    pad = stack - valid_shape_large.shape[0]
                    pad_shape_tuple = (block_size, other_dim) if block_rows else (other_dim, block_size)
                    pad_arr = jnp.broadcast_to(jnp.array(pad_shape_tuple, dtype=jnp.int32), (pad, 2))
                    valid_shape_large = jnp.concatenate([valid_shape_large, pad_arr])

                if pipeline_axis_name is not None:
                    # ensure vmapped inputs are constrained
                    valid_shape_large = _shard(valid_shape_large, PartitionSpec(pipeline_axis_name))
                    Ql_in = _shard(st.Ql, PartitionSpec(pipeline_axis_name))
                    Qr_in = _shard(st.Qr, PartitionSpec(pipeline_axis_name))
                    Ll_in = _shard(st.Ll, PartitionSpec(pipeline_axis_name))
                    Lr_in = _shard(st.Lr, PartitionSpec(pipeline_axis_name))
                else:
                    Ql_in, Qr_in, Ll_in, Lr_in = st.Ql, st.Qr, st.Ll, st.Lr

                Ql_new, Qr_new, Ll_new, Lr_new, Pg = vmap(
                    _precondition_2d_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None)
                )(keys, Ql_in, Qr_in, Ll_in, Lr_in, Gs, valid_shape_large, diag_left, diag_right, plr, noise_scale)
                if pipeline_axis_name is not None:
                    Pg = _shard(Pg, PartitionSpec(pipeline_axis_name))

                state["large"][leaf_idx] = st.replace(
                    Ql=_shard(otu.tree_cast(Ql_new, dtype), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Ql_new, dtype),
                    Qr=_shard(otu.tree_cast(Qr_new, dtype), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Qr_new, dtype),
                    Ll=_shard(otu.tree_cast(Ll_new, jnp.float32), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Ll_new, jnp.float32),
                    Lr=_shard(otu.tree_cast(Lr_new, jnp.float32), PartitionSpec(pipeline_axis_name)) if pipeline_axis_name is not None else otu.tree_cast(Lr_new, jnp.float32),
                )

                Pg = Pg[: (B * num_blocks_per_sample)]
                rec_samples = []
                start = 0
                for b in range(B):
                    nb = num_blocks_per_sample
                    rec = unblock_fn(Pg[start : start + nb], metas[b], block_size)
                    start += nb
                    rec_samples.append(rec)
                rec = jnp.stack(rec_samples, 0)
                leaves_u[leaf_idx] = jnp.reshape(rec, leaf.shape)

        # one_d path: sign(momentum)
        leaves_mupd, _ = jax.tree.flatten(mupd)
        for leaf_idx, (leaf, st) in enumerate(zip(leaves_u, perleaf_state)):
            if st.kind != KIND_ONE_D:
                continue  # not 1D
            B = st.B
            g = leaves_mupd[leaf_idx].astype(dtype)
            g2 = jnp.reshape(g, (B, -1))
            out = jnp.sign(g2)
            leaves_u[leaf_idx] = jnp.reshape(out, leaf.shape)

        precond_all = tdef_u.unflatten(leaves_u)

        if params_partition_specs is not None:
            precond_all = _shard(precond_all, params_partition_specs)

        # clip update RMS        
        precond_all = jax.tree.map(
            lambda g: g * (1.05 / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(g))), 1.05)), precond_all
        )

        # adam-style scale for compatibility
        if lr_style == "adam":
            precond_all = jax.tree.map(lambda g: g / jnp.array(5.0, g.dtype), precond_all)

        # re-box flax partitioned if needed
        if flax_partitioned and params is not None:
            flat_p, tdef_p = jax.tree.flatten(
                params, is_leaf=lambda g: hasattr(g, "replace_boxed") or isinstance(g, jax.ShapeDtypeStruct)
            )
            flat_g, _ = jax.tree.flatten(precond_all)
            if any(hasattr(p, "replace_boxed") for p in flat_p):
                precond_all = tdef_p.unflatten(
                    [p.replace_boxed(g) if hasattr(p, "replace_boxed") else g for p, g in zip(flat_p, flat_g)]
                )

        # persist
        state["count"] = step
        state["mu"] = mu
        return precond_all, state

    return base.GradientTransformation(init_fn, update_fn)


def quad(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    lr_style: Optional[str] = "adam",
    b1: float = 0.95,
    weight_decay: float = 0.1,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: Union[str, jnp.dtype] = jnp.bfloat16,
    scanned_layers: Optional[base.Params] = None,
    block_size: int = 256,
    pipeline_axis_name: Optional[str] = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: Optional[Union[PartitionSpec, List, Tuple, Dict]] = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    tx = [
        scale_by_quad(
            lr_style=lr_style,
            b1=b1,
            normalize_grads=normalize_grads,
            max_size_dense=max_size_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            dtype=dtype,
            scanned_layers=scanned_layers,
            block_size=block_size,
            pipeline_axis_name=pipeline_axis_name,
            pipeline_axis_size=pipeline_axis_size,
            params_partition_specs=params_partition_specs,
            noise_scale=noise_scale,
        )
    ]
    if weight_decay > 0.0:
        tx.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    tx.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*tx)


def get_opt_state_partition_specs(params, **quad_kwargs):
    _allowed = {
        "lr_style",
        "b1",
        "normalize_grads",
        "max_size_dense",
        "preconditioner_lr",
        "preconditioner_init_scale",
        "dtype",
        "scanned_layers",
        "block_size",
        "pipeline_axis_name",
        "pipeline_axis_size",
        "params_partition_specs",
        "noise_scale",
    }
    precond_kwargs = {k: v for k, v in quad_kwargs.items() if k in _allowed}
    # extract weight_decay for wrapping
    weight_decay = float(quad_kwargs.get("weight_decay", 0.0) or 0.0)
    tx = scale_by_quad(**precond_kwargs)
    state_shape = jax.eval_shape(tx.init, params)
    pipeline_axis_name = precond_kwargs.get("pipeline_axis_name", None)
    b1 = precond_kwargs.get("b1", 0.95)
    params_partition_specs = precond_kwargs.get("params_partition_specs", None)
    replicated = PartitionSpec()

    def _leading_axis_spec(ndim: int) -> PartitionSpec:
        if pipeline_axis_name is None or ndim == 0:
            return replicated
        return PartitionSpec(*([pipeline_axis_name] + [None] * (ndim - 1)))

    if b1 and b1 > 0:
        if params_partition_specs is not None:
            mu_specs = params_partition_specs
        else:

            def _param_spec(p):
                try:
                    return p.sharding.spec
                except Exception:
                    return replicated

            mu_specs = jax.tree.map(_param_spec, params)
    else:
        mu_specs = None

    def _to_specs(x, key_path: Tuple[Any, ...] = ()):
        if isinstance(x, jax.ShapeDtypeStruct):
            return _leading_axis_spec(x.ndim)
        if HAVE_FLAX and isinstance(x, LeafState):
            return x.replace(
                Ql=_to_specs(x.Ql),
                Qr=_to_specs(x.Qr),
                Ll=_to_specs(x.Ll),
                Lr=_to_specs(x.Lr),
            )
        if HAVE_FLAX and isinstance(x, DenseState):
            return x.replace(
                Ql=_to_specs(x.Ql),
                Qr=_to_specs(x.Qr),
                Ll=_to_specs(x.Ll),
                Lr=_to_specs(x.Lr),
                valid_rows=_to_specs(x.valid_rows),
                valid_cols=_to_specs(x.valid_cols),
            )
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k == "mu" and mu_specs is not None:
                    out[k] = mu_specs
                else:
                    out[k] = _to_specs(v, key_path + (k,))
            return out
        if isinstance(x, (list, tuple)):
            mapped = [_to_specs(v, key_path + (i,)) for i, v in enumerate(x)]
            return type(x)(mapped)
        return None

    precond_specs = _to_specs(state_shape)
    if weight_decay > 0.0:
        return (precond_specs, None, None)
    else:
        return (precond_specs, None)
