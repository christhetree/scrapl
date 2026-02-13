import functools
import logging
import os
from collections import defaultdict
from typing import Union, Optional, Iterable, List, Dict, Any, Callable, Literal

import hessian_eigenthings
import torch as tr
import torch.nn as nn
from hessian_eigenthings.operator import LambdaOperator
from torch import Tensor as T
from torch.nn import Parameter

from .single_path_jtfs import TimeFrequencyScrapl
from .util import ReadOnlyTensorDict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLoss(nn.Module):
    STATE_DICT_EXCLUDED_PREFIXES = [
        "jtfs.tensor",
    ]
    STATE_DICT_INCLUDED_ATTRS = [
        # Path related setup
        "path_counts",
        "unsampled_paths",
        # Sampling probs setup
        "probs",
        "loaded_probs_path",
        # Update probs setup
        "updated_path_indices",
        "all_log_vals",
        "all_log_probs",
    ]

    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        J_fr: int,
        Q_fr: int,
        T: Optional[Union[str, int]] = None,
        F: Optional[Union[str, int]] = None,
        p: int = 2,
        use_rho_log1p: bool = False,
        log1p_eps: float = 1e-3,  # TODO: what's a good default here?
        grad_mult: float = 1e8,
        use_p_adam: bool = True,
        use_p_saga: bool = True,
        sample_all_paths_first: bool = False,
        n_theta: int = 1,
        min_prob_frac: float = 0.0,
        probs_path: Optional[str] = None,
        eps: float = 1e-12,
        p_adam_b1: float = 0.9,
        p_adam_b2: float = 0.999,
        p_adam_eps: float = 1e-8,
    ):
        """Initializes a `SCRAPLLoss` module which contains an implementation of the
        "Scattering Transform with Random Paths for Machine Learning" (SCRAPL) algorithm
        for the joint time-frequency scattering transform (JTFS) in PyTorch.
        For documentation, examples, hyperparameters, and best practices, please visit:
        https://github.com/christhetree/scrapl
        For more information about the JTFS and the `J`, `Q1`, `Q2`, `J_fr`, `Q_fr`,
        `T`, and `F` hyperparameters, please visit:
        https://www.kymat.io/ismir23-tutorial/intro.html

        Args:
            shape (int): The length of the input signal (number of samples).
            J (int): Number of octaves in the JTFS (1st and 2nd order temporal filters)
            Q1 (int): Wavelets per octave in the JTFS (1st order temporal filters)
            Q2 (int): Wavelets per octave in the JTFS (2nd order temporal filters)
            J_fr (int): Number of octaves in the JTFS (2nd order frequential filters)
            Q_fr (int): Wavelets per octave in the JTFS (2nd order frequential filters)
            T (Optional[Union[str, int]]): Temporal averaging size in samples of the
                JTFS. If 'global', averages over the entire signal. If 'None', averages
                over J**2 samples.
                Defaults to None.
            F (Optional[Union[str, int]]): Frequential averaging size in frequency
                bins of the JTFS. If 'global', averages over all bins. If 'None',
                averages over J_fr * Q_fr bins.
                Defaults to None.
            p (int, optional): The order of the norm used for the distance calculation.
                Defaults to 2 (Euclidean norm).
            use_rho_log1p (bool, optional): If True, applies log1p scaling to the
                scattering coefficients (log(1 + x / log1p_eps)) before computing the
                distance. If True, `grad_mult` is no longer necessary and can be set to
                a value of 1.
                Defaults to False.
            log1p_eps (float, optional): The epsilon value used in the log1p scaling.
                Defaults to 1e-3.
            grad_mult (float, optional): A scalar multiplier applied to gradients to
                prevent JTFS precision errors when squaring gradient values in commonly
                used optimizers like Adam. See https://hal.science/hal-05124224v1 for
                more information. When `use_rho_log1p` is True, `grad_mult` is no
                longer necessary and can be set to a value of 1. When `grad_mult` is
                not 1, `attach_params()` must be called with the model parameters
                being optimized for this to have an effect.
                Defaults to 1e8.
            use_p_adam (bool, optional): If True, enables the $\mathcal{P}$-Adam
                algorithm. When True, `attach_params()` must be called before training
                with the model parameters being optimized and vanilla stochastic
                gradient descent (SGD) with no momentum and optional weight decay
                should be used as the downstream optimizer.
                Defaults to True.
            use_p_saga (bool, optional): If True, enables the $\mathcal{P}$-SAGA
                algorithm. When True, `attach_params()` must be called before training
                with the model parameters being optimized and vanilla stochastic
                gradient descent (SGD) with no momentum and optional weight decay
                should be used as the downstream optimizer.
                Defaults to True.
            sample_all_paths_first (bool, optional): If True, forces the sampler to
                visit every possible scattering path once in order before switching to
                the internal path sampling distribution.
                Defaults to False.
            n_theta (int, optional): Given an encoder and decoder (synth)
                self-supervised training architecture, the number of encoder output /
                decoder (synth) input parameters $\theta_{\mathrm{synth}}$. This is
                only required for the importance sampling heuristic ($\theta$-IS) and
                calling the `warmup_lc_hvp()` method before training.
                Defaults to 1.
            min_prob_frac (float, optional): When using $\theta$-IS$, the minimum
                fraction of the uniform sampling probability assigned to any path,
                ensuring no path has zero probability of being sampled.
                Defaults to 0.0.
            probs_path (Optional[str], optional): File path to a `.pt` file containing
                pre-computed sampling probabilities for the scattering paths.
                Defaults to None.
            eps (float, optional): A small value for numerical stability in probability
                calculations.
                Defaults to 1e-12.
            p_adam_b1 (float, optional): $\beta_1$ Adam hyperparameter for the internal
                $\mathcal{P}$-Adam algorithm.
                Defaults to 0.9.
            p_adam_b2 (float, optional): $\beta_2$ Adam hyperparameter for the internal
                $\mathcal{P}$-Adam algorithm.
                Defaults to 0.999.
            p_adam_eps (float, optional): $\epsilon$ Adam hyperparameter for the
                internal $\mathcal{P}$-Adam algorithm.
                Defaults to 1e-8.

        Raises:
            AssertionError: If `grad_mult <= 1.0` when `use_p_adam` is True and
                `use_rho_log1p` is False (to avoid precision errors).
            AssertionError: If `n_theta < 1`.
            AssertionError: If `min_prob_frac` is not in [0, 1).
        """
        super().__init__()
        self.p = p
        self.use_rho_log1p = use_rho_log1p
        self.log1p_eps = log1p_eps
        self.grad_mult = grad_mult
        if use_p_adam and not use_rho_log1p:
            assert (
                grad_mult > 1.0
            ), "Using p_adam with no log1p requires a grad multiplier to avoid float precision errors"
        self.use_p_adam = use_p_adam
        self.use_p_saga = use_p_saga
        self.sample_all_paths_first = sample_all_paths_first
        assert n_theta >= 1
        self.n_theta = n_theta
        assert 0 <= min_prob_frac < 1.0
        self.min_prob_frac = min_prob_frac
        self.eps = eps
        self.p_adam_b1 = p_adam_b1
        self.p_adam_b2 = p_adam_b2
        self.p_adam_eps = p_adam_eps

        # Path related setup
        self.jtfs = TimeFrequencyScrapl(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
        )
        self.meta = self.jtfs.meta()
        if len(self.meta["key"]) != len(self.meta["order"]):
            self.meta["key"] = self.meta["key"][1:]
        # TODO: add first order only keys too?
        self.scrapl_keys = [key for key in self.meta["key"] if len(key) == 2]
        self.n_paths = len(self.scrapl_keys)
        self.unif_prob = 1.0 / self.n_paths
        log.info(
            f"SCRAPLLoss:\n"
            f"J={J}, Q1={Q1}, Q2={Q2}, Jfr={J_fr}, Qfr={Q_fr}, T={T}, F={F}\n"
            f"use_rho_log1p          = {use_rho_log1p}, eps = {log1p_eps}\n"
            f"grad_mult              = {grad_mult}\n"
            f"use_p_adam             = {use_p_adam}\n"
            f"use_p_saga             = {use_p_saga}\n"
            f"sample_all_paths_first = {sample_all_paths_first}\n"
            f"n_theta                = {n_theta}\n"
            f"min_prob_frac          = {min_prob_frac}\n"
            f"number of SCRAPL paths = {self.n_paths}\n"
            f"unif_prob              = {self.unif_prob:.8f}\n"
        )
        self.curr_path_idx = None
        self.path_counts = defaultdict(int)
        self.rand_gen = tr.Generator(device="cpu")
        # This is only used if sample_all_paths_first is True
        self.unsampled_paths = list(range(self.n_paths))

        # Grad related setup
        self.attached_params = []
        self.p_adam_m = {}
        self.p_adam_v = {}
        self.p_adam_t = defaultdict(lambda: defaultdict(int))
        self.scrapl_t = 0
        self.prev_path_grads = {}
        self.hook_handles = []

        # Sampling probs setup
        # We keep probs on the CPU to avoid GPU and CPU seed discrepancies
        self.loaded_probs_path = None
        if probs_path is None:
            self.probs = tr.full((self.n_paths,), self.unif_prob, dtype=tr.double)
        else:
            self.load_probs(probs_path)

        # Update probs setup
        self.log_eps = tr.tensor(self.eps, dtype=tr.double).log()
        self.log_unif_prob = tr.tensor(self.unif_prob, dtype=tr.double).log()
        self.min_prob = self.unif_prob * self.min_prob_frac
        self.log_1m_min_prob_frac = tr.tensor(
            1.0 - self.min_prob_frac, dtype=tr.double
        ).log()
        self.updated_path_indices = defaultdict(set)
        self.all_log_vals = tr.full(
            (self.n_theta, self.n_paths), self.log_eps, dtype=tr.double
        )
        self.all_log_probs = tr.full(
            (
                self.n_theta,
                self.n_paths,
            ),
            self.log_unif_prob,
            dtype=tr.double,
        )

        # Warmup setup
        theta_eye = tr.eye(self.n_theta)
        self.register_buffer("theta_eye", theta_eye, persistent=False)

        # Check probs
        self._check_probs()

    def load_probs(self, probs_path: str) -> None:
        assert os.path.isfile(probs_path)
        probs = tr.load(probs_path).double()
        assert probs.shape == (
            self.n_paths,
        ), f"probs.shape = {probs.shape}, expected {(self.n_paths,)}"
        log.info(
            f"Loading probs from:\n\t{probs_path}\n\tmin prob = {probs.min():.6f}, "
            f"max prob = {probs.max():.6f}"
        )
        self.probs = probs
        self.loaded_probs_path = probs_path
        self._check_probs()

    def load_probs_from_warmup_dir(
        self, warmup_dir: str, agg: Literal["none", "mean", "max", "med"] = "none"
    ) -> None:
        assert os.path.isdir(warmup_dir)
        self.clear()
        for path_idx in range(self.n_paths):
            vals_path = os.path.join(warmup_dir, f"vals_{path_idx}.pt")
            if not os.path.isfile(vals_path):
                log.warning(
                    f"Warmup vals for path index {path_idx} do not exist, skipping"
                )
                continue
            vals = tr.load(vals_path)
            self._aggregate_vals_and_update_probs(vals, path_idx, agg=agg)
        self.loaded_probs_path = warmup_dir
        log.info(
            f"Loading probs from directory:\n\t{os.path.abspath(warmup_dir)}\n\t"
            f"min prob = {self.probs.min():.6f}, max prob = {self.probs.max():.6f}"
        )
        self._check_probs()

    def _clear_grad_data(self) -> None:
        del self.p_adam_m
        self.p_adam_m = {}
        del self.p_adam_v
        self.p_adam_v = {}
        self.p_adam_t = defaultdict(lambda: defaultdict(int))
        self.scrapl_t = 0
        del self.prev_path_grads
        self.prev_path_grads = {}
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.attached_params = []

    def clear(self) -> None:
        # Path related setup
        self.curr_path_idx = None
        self.path_counts.clear()
        self.unsampled_paths = list(range(self.n_paths))
        # Grad related setup
        self._clear_grad_data()
        # Update probs setup
        self.updated_path_indices.clear()
        self.all_log_vals.fill_(self.log_eps)
        self.all_log_probs.fill_(self.log_unif_prob)
        self._recompute_probs()
        self.loaded_probs_path = None
        log.info(f"Cleared state")

    def state_dict(self, *args, **kwargs) -> Dict[str, T]:
        # TODO: support resuming training with grad hooks
        state_dict = super().state_dict(*args, **kwargs)
        global_prefix = kwargs.get("prefix", "")
        excluded_keys = []
        for prefix in self.STATE_DICT_EXCLUDED_PREFIXES:
            for k in state_dict:
                if k.startswith(f"{global_prefix}{prefix}"):
                    excluded_keys.append(k)
        for k in excluded_keys:
            del state_dict[k]
        for attr in self.STATE_DICT_INCLUDED_ATTRS:
            assert attr not in state_dict
            val = getattr(self, attr)
            # Attempt to copy or clone the val to avoid potential issues with
            # references to mutable objects in the state dict
            try:
                val = val.copy()
            except AttributeError:
                pass
            try:
                val = val.clone()
            except AttributeError:
                pass
            state_dict[f"{global_prefix}{attr}"] = val
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, T], *args, **kwargs) -> None:
        global_prefix = kwargs.get("prefix", "")
        for attr in self.STATE_DICT_INCLUDED_ATTRS:
            assert f"{global_prefix}{attr}" in state_dict
            setattr(self, attr, state_dict[f"{global_prefix}{attr}"])
        kwargs["strict"] = False  # TODO: is there a better way to do this?
        super().load_state_dict(state_dict, *args, **kwargs)

    def attach_params(self, params: Iterable[T]) -> None:
        n_prev_attached_params = len(self.attached_params)
        self._clear_grad_data()
        self.attached_params = list(params)
        p_adam_m = {}
        p_adam_v = {}
        prev_path_grads = {}
        total_n_weights = 0
        for idx, p in enumerate(self.attached_params):
            p_adam_m[idx] = tr.zeros((self.n_paths, *p.shape))
            p_adam_v[idx] = tr.zeros((self.n_paths, *p.shape))
            prev_path_grads[idx] = tr.zeros((self.n_paths, *p.shape))
            handle = p.register_hook(functools.partial(self.grad_hook, param_idx=idx))
            self.hook_handles.append(handle)
            total_n_weights += p.numel()
        del self.p_adam_m
        del self.p_adam_v
        del self.prev_path_grads
        self.register_module("p_adam_m", ReadOnlyTensorDict(p_adam_m, persistent=False))
        self.register_module("p_adam_v", ReadOnlyTensorDict(p_adam_v, persistent=False))
        self.register_module(
            "prev_path_grads", ReadOnlyTensorDict(prev_path_grads, persistent=False)
        )
        if not self.attached_params:
            log.info(f"Detached {n_prev_attached_params} parameter tensors")
        else:
            log.info(
                f"Attached {len(self.prev_path_grads)} parameter tensors "
                f"({total_n_weights} weights)"
            )

    def grad_hook(self, grad: T, param_idx: int) -> T:
        # TODO: check if this works
        if not self.training:
            return grad

        if self.grad_mult != 1.0:
            grad *= self.grad_mult

        assert self.curr_path_idx is not None
        path_idx = self.curr_path_idx
        curr_t = self.scrapl_t + 1

        if self.use_p_adam:
            prev_m_s = self.p_adam_m[param_idx]
            prev_v_s = self.p_adam_v[param_idx]
            prev_t_s = self.p_adam_t[param_idx]
            prev_m = prev_m_s[path_idx]
            prev_v = prev_v_s[path_idx]
            prev_t = prev_t_s[path_idx]
            prev_t_norm = prev_t / self.n_paths
            t_norm = curr_t / self.n_paths

            grad, m, v = self.adam_grad_norm_cont(
                grad,
                prev_m,
                prev_v,
                t_norm,
                prev_t_norm,
                b1=self.p_adam_b1,
                b2=self.p_adam_b2,
                eps=self.p_adam_eps,
            )
            prev_m_s[path_idx] = m
            prev_v_s[path_idx] = v
            prev_t_s[path_idx] = curr_t

        if self.use_p_saga:
            n_paths_seen = len(self.path_counts)
            # Calculate previous average grad
            prev_path_grads = self.prev_path_grads[param_idx]
            prev_avg_grad = prev_path_grads.sum(dim=0) / max(1, n_paths_seen - 1)
            # Get prev path grad
            prev_path_grad = prev_path_grads[path_idx, ...]
            # Calculate SAGA grad
            saga_grad = grad - prev_path_grad + prev_avg_grad
            # Update current path grad
            prev_path_grads[path_idx, ...] = grad
            # Update grad
            grad = saga_grad

        return grad

    def sample_path(self, seed: Optional[int] = None) -> int:
        if seed is None:
            rand_gen = None
        else:
            rand_gen = self.rand_gen
            rand_gen.manual_seed(seed)
        if (
            self.sample_all_paths_first
            and len(self.path_counts) < self.n_paths
            and self.unsampled_paths
        ):
            unsampled_idx = tr.randint(
                0, len(self.unsampled_paths), (1,), generator=rand_gen
            ).item()
            # TODO: will this pop during validation and test?
            path_idx = self.unsampled_paths.pop(unsampled_idx)
        else:
            self.check_probs(self.probs, self.n_paths, self.eps, self.min_prob)
            path_idx = tr.multinomial(
                self.probs, num_samples=1, generator=rand_gen
            ).item()
        return path_idx

    def calc_dist(self, x: T, x_target: T, path_idx: int) -> (T, T, T):
        n2, n_fr = self.scrapl_keys[path_idx]
        Sx = self.jtfs.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        if self.use_rho_log1p:
            Sx = tr.log1p(Sx / self.log1p_eps)
        Sx_target = self.jtfs.scattering_singlepath(x_target, n2, n_fr)
        Sx_target = Sx_target["coef"].squeeze(-1)
        if self.use_rho_log1p:
            Sx_target = tr.log1p(Sx_target / self.log1p_eps)
        diff = Sx_target - Sx
        dist = tr.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
        # TODO: add reduction option
        dist = tr.mean(dist)
        return dist, Sx, Sx_target

    def forward(
        self,
        x: T,
        x_target: T,
        seed: Optional[int] = None,
        path_idx: Optional[int] = None,
    ) -> T:
        assert x.ndim == x_target.ndim == 3
        bs, n_ch, n_samples = x.size()
        x = x.view(-1, 1, n_samples)
        x_target = x_target.view(-1, 1, n_samples)
        assert x.size(1) == x_target.size(1) == 1
        if path_idx is None:
            path_idx = self.sample_path(seed)
            log.debug(f"Sample path_idx = {path_idx}")
        else:
            log.debug(f"Using specified path_idx = {path_idx}")
            assert 0 <= path_idx < self.n_paths
            if seed is not None:
                log.warning(f"seed is ignored when path_idx is specified")
        dist, _, _ = self.calc_dist(x, x_target, path_idx)

        self.curr_path_idx = path_idx
        if self.training:  # TODO: check if this works
            self.path_counts[path_idx] += 1
            self.scrapl_t += 1
            # if self.grad_mult != 1.0 or self.use_p_adam or self.use_p_saga:
            #     if not self.attached_params:
            #         log.warning(
            #             f"Parameters are not attached, but grad_mult "
            #             f"({self.grad_mult}), use_p_adam ({self.use_p_adam}), or "
            #             f"use_p_saga ({self.use_p_saga}) are enabled."
            #         )
        return dist

    # Update prob methods ==============================================================
    def update_prob(self, path_idx: int, val: float | T, theta_idx: int = 0) -> None:
        assert self.loaded_probs_path is None, (
            f"Cannot update probs if external probs have been loaded "
            f"({self.loaded_probs_path})"
        )
        assert 0 <= path_idx < self.n_paths
        assert 0 <= theta_idx < self.n_theta
        assert val >= 0.0
        if isinstance(val, float):
            val = tr.tensor(val, dtype=tr.double)
        val = val.cpu()
        # Keep track of which and how many paths have been updated
        self.updated_path_indices[theta_idx].add(path_idx)
        updated_indices = list(self.updated_path_indices[theta_idx])
        n_updated = len(updated_indices)
        # Update log vals
        log_vals = self.all_log_vals[theta_idx]
        log_val = val.double().clamp(min=self.eps).log()
        log_vals[path_idx] = log_val
        # Update log probs
        if n_updated < 2:
            # Use init probs since there are not enough updated paths
            return
        elif n_updated == self.n_paths:
            # Use all updated log vals to calculate log probs
            log_probs = self.calc_log_probs(log_vals)
        else:
            # Use a balance between updated log vals and init probs
            log_probs = self.all_log_probs[theta_idx]
            log_probs[updated_indices] = self.log_eps
            updated_frac = tr.tensor(n_updated / self.n_paths, dtype=tr.double)
            stale_frac = 1.0 - updated_frac
            updated_probs = self.calc_log_probs(log_vals) + updated_frac.log()
            stale_probs = self.calc_log_probs(log_probs) + stale_frac.log()
            self.all_log_probs[theta_idx] = stale_probs
            self.all_log_probs[theta_idx][updated_indices] = updated_probs[
                updated_indices
            ]
            log_probs = self.calc_log_probs(self.all_log_probs[theta_idx])

        # Ensure min_prob_frac is enforced
        log_probs += self.log_1m_min_prob_frac
        # Update all log probs
        self.all_log_probs[theta_idx] = log_probs
        # Update probs
        self._recompute_probs()

    def _recompute_probs(self) -> None:
        all_probs = self.all_log_probs.exp() + self.min_prob
        probs = all_probs.mean(dim=0)
        self.probs = probs
        self._check_probs()

    def _check_probs(self) -> None:
        self.check_probs(self.probs, self.n_paths, self.eps, self.min_prob)
        all_probs = self.all_log_probs.exp()
        all_vals = self.all_log_vals.exp()
        for theta_idx in range(self.n_theta):
            probs = all_probs[theta_idx, :]
            vals = all_vals[theta_idx, :]
            updated_indices = list(self.updated_path_indices[theta_idx])
            self.check_probs(
                probs, self.n_paths, self.eps, self.min_prob, vals, updated_indices
            )

    # Warmup methods ===================================================================
    def _calc_batch_theta_param_grad(
        self,
        path_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: Dict[str, Any],
        params: List[Parameter],
        theta_idx: Optional[int] = None,
        seed: Optional[int] = None,
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> T:
        if synth_fn_kwargs is None:
            synth_fn_kwargs = {}
        assert "x" in theta_fn_kwargs, "x must be provided in theta_fn_kwargs"
        assert params, "params must not be empty"
        assert all(p.grad is None for p in params), "params must have no grad"
        if theta_idx is not None:
            assert 0 <= theta_idx < self.n_theta
        theta_hat = theta_fn(**theta_fn_kwargs)
        assert theta_hat.ndim == 2
        assert theta_hat.size(1) == self.n_theta
        bs = theta_hat.size(0)
        x_hat = synth_fn(theta_hat, **synth_fn_kwargs)
        x = theta_fn_kwargs["x"]
        loss = self.forward(x, x_hat, seed=seed, path_idx=path_idx)

        theta_grad = tr.autograd.grad(
            loss, theta_hat, create_graph=True, materialize_grads=True
        )[0]
        if theta_idx is None:
            # Expand theta_grad to vectorize individual theta grad calculations
            theta_grad_ex = theta_grad.unsqueeze(0).expand(self.n_theta, -1, -1)
            theta_eye_ex = self.theta_eye.unsqueeze(1).expand(-1, bs, -1)
            theta_grad_ex = theta_grad_ex * theta_eye_ex
            is_grads_batched = True
        else:
            # Zero out all but the specified theta grad
            mask = tr.zeros_like(theta_grad)
            mask[:, theta_idx] = 1.0
            theta_grad_ex = theta_grad * mask
            is_grads_batched = False
        # Calculate maybe vectorized param grad
        theta_param_grads = tr.autograd.grad(
            theta_hat,
            params,
            grad_outputs=theta_grad_ex,
            create_graph=True,
            materialize_grads=True,
            is_grads_batched=is_grads_batched,
        )
        if theta_idx is None:
            theta_param_grad = tr.cat(
                [g.view(self.n_theta, -1) for g in theta_param_grads], dim=1
            )
        else:
            theta_param_grad = tr.cat([g.view(-1) for g in theta_param_grads])
        return theta_param_grad

    def _calc_param_hvp(
        self,
        tangent: T,
        param_grad: T,
        params: List[Parameter],
        retain_graph: bool = False,
    ) -> T:
        assert tangent.ndim == 1
        assert param_grad.shape == tangent.shape
        assert all(p.grad is None for p in params), "params must have no grad"
        param_hvps = tr.autograd.grad(
            param_grad,
            params,
            grad_outputs=tangent,
            materialize_grads=True,
            retain_graph=retain_graph,
        )
        param_hvp = tr.cat([g.contiguous().view(-1) for g in param_hvps])
        return param_hvp

    def _calc_largest_eig(
        self,
        param_grad: T,
        params: List[Parameter],
        n_iter: int = 20,
    ) -> T:
        apply_fn = functools.partial(
            self._calc_param_hvp,
            param_grad=param_grad,
            params=params,
            retain_graph=True,
        )
        size = param_grad.size(0)
        hvp_op = LambdaOperator(apply_fn, size)
        use_gpu = param_grad.is_cuda
        eigs, _ = hessian_eigenthings.deflated_power_iteration(
            operator=hvp_op,
            num_eigenthings=1,
            power_iter_steps=n_iter,
            to_numpy=False,
            use_gpu=use_gpu,
        )
        eig1 = tr.from_numpy(eigs.copy()).float()
        return eig1

    def calc_theta_eigs(
        self,
        path_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: Dict[str, Any],
        params: List[Parameter],
        seed: Optional[int] = None,
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
        n_iter: int = 20,
    ) -> T:
        theta_param_grad = self._calc_batch_theta_param_grad(
            path_idx,
            theta_fn,
            synth_fn,
            theta_fn_kwargs,
            params,
            seed=seed,
            synth_fn_kwargs=synth_fn_kwargs,
        )
        theta_eigs = []
        for theta_idx in range(self.n_theta):
            param_grad = theta_param_grad[theta_idx, :]
            eig1 = self._calc_largest_eig(param_grad, params, n_iter=n_iter)
            theta_eigs.append(eig1)
        theta_eigs = tr.cat(theta_eigs, dim=0)
        return theta_eigs

    def _calc_param_hvp_multibatch(
        self,
        tangent: T,
        path_idx: int,
        theta_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> T:
        if synth_fn_kwargs is None:
            assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
            synth_fn_kwargs = [None] * len(theta_fn_kwargs)
        else:
            assert len(synth_fn_kwargs) == len(theta_fn_kwargs), (
                f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
                f"len(synth_fn_kwargs) ({len(synth_fn_kwargs)})"
            )
        param_hvp = None
        for curr_theta_fn_kwargs, curr_synth_fn_kwargs in zip(
            theta_fn_kwargs, synth_fn_kwargs
        ):
            curr_param_grad = self._calc_batch_theta_param_grad(
                path_idx,
                theta_fn,
                synth_fn,
                curr_theta_fn_kwargs,
                params,
                theta_idx=theta_idx,
                synth_fn_kwargs=curr_synth_fn_kwargs,
            )
            curr_param_hvp = self._calc_param_hvp(
                tangent, curr_param_grad, params, retain_graph=False
            )
            # TODO: should we average here?
            if param_hvp is None:
                param_hvp = curr_param_hvp
            else:
                param_hvp += curr_param_hvp
        return param_hvp

    def _calc_theta_largest_eig_multibatch(
        self,
        path_idx: int,
        theta_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
    ) -> T:
        apply_fn = functools.partial(
            self._calc_param_hvp_multibatch,
            path_idx=path_idx,
            theta_idx=theta_idx,
            theta_fn=theta_fn,
            synth_fn=synth_fn,
            theta_fn_kwargs=theta_fn_kwargs,
            params=params,
            synth_fn_kwargs=synth_fn_kwargs,
        )
        size = sum(p.numel() for p in params)
        hvp_op = LambdaOperator(apply_fn, size)
        use_gpu = params[0].is_cuda
        eigs, _ = hessian_eigenthings.deflated_power_iteration(
            operator=hvp_op,
            num_eigenthings=1,
            power_iter_steps=n_iter,
            to_numpy=False,
            use_gpu=use_gpu,
        )
        eig1 = tr.from_numpy(eigs.copy()).float()
        return eig1

    def calc_theta_eigs_multibatch(
        self,
        path_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
    ) -> T:
        theta_eigs = []
        for theta_idx in range(self.n_theta):
            eig1 = self._calc_theta_largest_eig_multibatch(
                path_idx,
                theta_idx,
                theta_fn,
                synth_fn,
                theta_fn_kwargs,
                params,
                synth_fn_kwargs=synth_fn_kwargs,
                n_iter=n_iter,
            )
            theta_eigs.append(eig1)
        theta_eigs = tr.cat(theta_eigs, dim=0)
        return theta_eigs

    def _aggregate_vals_and_update_probs(
        self,
        vals: T,
        path_idx: int,
        agg: Literal["none", "mean", "max", "med"] = "none",
    ) -> None:
        assert vals.ndim == 2
        assert vals.size(1) == self.n_theta
        if agg == "none":
            assert vals.size(0) == 1
            vals = vals[0]
        elif agg == "mean":
            vals = vals.mean(dim=0)
        elif agg == "max":
            vals = vals.max(dim=0).values
        elif agg == "med":
            vals = vals.median(dim=0).values
        else:
            raise ValueError(f"Invalid agg = {agg}")
        # Update the probs for each theta
        for theta_idx in range(self.n_theta):
            val = vals[theta_idx]
            self.update_prob(path_idx, val, theta_idx)

    def warmup_lc_hvp(
        self,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        start_path_idx: int = 0,
        end_path_idx: Optional[int] = None,
        save_dir: Optional[str] = "scrapl_warmup",
        n_iter: int = 20,
        agg: Literal["none", "mean", "max", "med"] = "none",
        force_multibatch: bool = False,
    ) -> None:
        """
        Executes the $\theta$-Importance Sampling ($\\theta$-IS) warmup procedure to
        initialize a non-uniform scattering path sampling probability distribution
        based on the local curvature of the loss landscape. For more information please
        visit: https://github.com/christhetree/scrapl/#importance-sampling-warmup

        **Parallelization:** To speed up warmup, this method can be run on multiple
        GPUs simultaneously by assigning disjoint [`start_path_idx`, `end_path_idx`)
        ranges to each process and providing a `save_dir`. After all processes finish,
        use `load_probs_from_warmup_dir` to load the aggregated path sampling
        probability distribution into a `SCRAPLLoss` instance.

        Args:
            theta_fn (Callable[..., T]): The encoder function. It must accept arguments
                provided in `theta_fn_kwargs` and return a tensor
                $\theta_{\mathrm{synth}}$ of shape `(batch_size, n_theta)`. This
                function should be deterministic during warmup, but can be
                non-deterministic otherwise.
            synth_fn (Callable[[T, ...], T]): The decoder (synthesizer) function.
                It must accept the $\theta_{\mathrm{synth}}$ tensor output by
                `theta_fn` (and optional `synth_fn_kwargs`) and return a signal tensor
                `x_hat` of shape `(n_batches, n_ch, n_samples)`. This function should
                be deterministic during warmup, but can be non-deterministic otherwise.
            theta_fn_kwargs (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary contains a batch of input arguments for `theta_fn`. One of
                these arguments must be the input signal `x` of shape
                `(n_batches, n_ch, n_samples)`. The length of this list determines the
                number of batches used for the loss landscape curvature estimation and
                is a tradeoff between computational cost and curvature estimation
                accuracy.
            params (List[Parameter]): A list of encoder parameters (learnable weights)
                involved in the computation of $\theta_{\mathrm{synth}}$. These
                parameters must have no prior gradients (i.e. `p.grad is None` for all
                `p` in `params`) before calling this method.
            synth_fn_kwargs (Optional[List[Dict[str, Any]]], optional): A list of
                dictionaries corresponding to `theta_fn_kwargs`, containing additional
                arguments for `synth_fn`. If provided, must have the same length as
                `theta_fn_kwargs`.
                Defaults to None.
            start_path_idx (int, optional): The starting index of the scattering paths
                to compute. Used for parallelizing the warmup across multiple devices.
                Defaults to 0.
            end_path_idx (Optional[int], optional): The ending index (exclusive) of
                the scattering paths to compute. If None, defaults to `self.n_paths`.
                Defaults to None.
            save_dir (Optional[str], optional): Directory path where intermediate
                eigenvalues (`vals_{path_idx}.pt`) and the final path sampling
                probability distribution (`probs.pt`) will be saved.
                Defaults to "scrapl_warmup".
            n_iter (int, optional): The maximum number of power iteration steps used to
                approximate the largest eigenvalue of the Hessian. Higher values
                increase precision at the cost of computation time.
                Defaults to 20.
            agg (Literal["none", "mean", "max", "med"], optional): The aggregation
                strategy if `params` are split into individual tensors. Usually kept as
                "none" to aggregate across the full parameter set provided unless it
                does not fit in GPU memory, in which case curvature is estimated for
                each parameter tensor separately and then aggregated at the end. If you
                are seeing "out of memory" errors, try switching this to "med" to
                reduce memory usage at the cost of computation time.
                Defaults to "none".
            force_multibatch (bool, optional): Debugging tool. If True, forces the
                calculation to use multibatch logic even if `theta_fn_kwargs` contains
                only a single batch.
                Defaults to False.

        Returns:
            None: The method updates `self.probs` in-place and saves results to
            `save_dir` if it is not None.

        Raises:
            AssertionError: If `params` is empty or contains parameters with existing
                            gradients.
            AssertionError: If params have already been attached to the `SCRAPLLoss`
                            instance.
            AssertionError: If `theta_fn_kwargs` is empty or if `synth_fn_kwargs` is
                            provided and has a different length than `theta_fn_kwargs`.
        """
        assert params, "params must not be empty"
        assert all(
            not p.grad for p in params
        ), "params must have no previous gradients before warmup"
        assert not self.attached_params, "Parameters cannot be attached during warmup"
        assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
        if synth_fn_kwargs is not None:
            assert len(synth_fn_kwargs) == len(theta_fn_kwargs), (
                f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
                f"len(synth_fn_kwargs) ({len(synth_fn_kwargs)})"
            )
        if end_path_idx is None:
            end_path_idx = self.n_paths
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Check determinism
        theta_fn_kwargs_batch = theta_fn_kwargs[0]
        synth_fn_kwargs_batch = {}
        if synth_fn_kwargs is not None:
            synth_fn_kwargs_batch = synth_fn_kwargs[0]
        self.check_is_deterministic(
            theta_fn, theta_fn_kwargs_batch, synth_fn, synth_fn_kwargs_batch
        )

        # Determine whether to use multibatch or not
        is_multibatch = force_multibatch or len(theta_fn_kwargs) > 1
        log.info(
            f"Starting warmup_lc_hvp with agg = {agg} for {len(params)} "
            f"parameter(s) and {len(theta_fn_kwargs)} batch(es), {n_iter} iter "
            f"(multibatch = {is_multibatch})"
        )
        if is_multibatch:
            calc_theta_eigs_fn = self.calc_theta_eigs_multibatch
        else:
            calc_theta_eigs_fn = self.calc_theta_eigs
            theta_fn_kwargs = theta_fn_kwargs[0]
            if synth_fn_kwargs is not None:
                synth_fn_kwargs = synth_fn_kwargs[0]
        # Separate the params for separate computations if aggregating LCs across them
        if agg == "none":
            param_groups = [params]
        else:
            param_groups = [[p] for p in params]
        # Compute the theta LCs for each path
        for path_idx in range(start_path_idx, end_path_idx):
            vals = []
            for param_group in param_groups:
                curr_vals = calc_theta_eigs_fn(
                    path_idx=path_idx,
                    theta_fn=theta_fn,
                    synth_fn=synth_fn,
                    theta_fn_kwargs=theta_fn_kwargs,
                    params=param_group,
                    synth_fn_kwargs=synth_fn_kwargs,
                    n_iter=n_iter,
                )
                # TODO: double check this
                curr_vals = curr_vals.abs()
                vals.append(curr_vals)
                log.info(f"path_idx = {path_idx}, curr_vals = {curr_vals}")
            # Aggregate the theta LCs across all params
            vals = tr.stack(vals, dim=0)

            # Save intermediate results for parallel computation
            if save_dir is not None:
                save_path = os.path.abspath(
                    os.path.join(save_dir, f"vals_{path_idx}.pt")
                )
                assert not os.path.isfile(
                    save_path
                ), f"save_path {save_path} already exists"
                tr.save(vals.detach().cpu(), save_path)

            # Update the sampling probs based on the theta LCs
            self._aggregate_vals_and_update_probs(vals, path_idx, agg=agg)

        # Only save the final probs if we have completed the warmup for all paths
        if (
            save_dir is not None
            and start_path_idx == 0
            and end_path_idx == self.n_paths
        ):
            save_path = os.path.abspath(os.path.join(save_dir, f"probs.pt"))
            log.info(f"Saving warmup SCRAPL sampling probabilities to {save_path}")
            tr.save(self.probs.cpu(), save_path)

    # Static methods ===================================================================
    @staticmethod
    def check_probs(
        probs: T,
        n_paths: int,
        eps: float,
        min_prob: float = 0.0,
        vals: Optional[T] = None,
        updated_indices: Optional[List[int]] = None,
    ) -> None:
        assert probs.shape == (n_paths,)
        assert tr.allclose(
            probs.sum(), tr.tensor(1.0, dtype=tr.double), atol=eps
        ), f"self.probs.sum() = {probs.sum()}"
        assert probs.min() >= min_prob - eps, f"probs.min() = {probs.min()}"

        if vals is None or len(updated_indices) < 2:
            return

        assert vals.shape == (n_paths,)
        assert vals.min() >= eps
        vals = vals[updated_indices]
        val_ratios = vals / vals.min()
        raw_probs = probs[updated_indices] - min_prob
        assert raw_probs.min() > 0.0
        prob_ratios = raw_probs / raw_probs.min()
        assert tr.allclose(
            val_ratios, prob_ratios, atol=1e-3, rtol=1e-3, equal_nan=True
        ), f"val_ratios = {val_ratios}, prob_ratios = {prob_ratios}"

    @staticmethod
    def calc_log_probs(log_vals: T) -> T:
        assert log_vals.ndim == 1
        log_probs = log_vals - tr.logsumexp(log_vals, dim=0)
        return log_probs

    @staticmethod
    def adam_grad_norm_cont(
        grad: T,
        prev_m: T,
        prev_v: T,
        t: float,
        prev_t: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> (T, T, T):
        delta_t = t - prev_t
        eff_b1 = b1**delta_t
        eff_b2 = b2**delta_t
        m = eff_b1 * prev_m + (1 - eff_b1) * grad
        v = eff_b2 * prev_v + (1 - eff_b2) * grad**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        grad_hat = m_hat / (T.sqrt(v_hat) + eps)
        return grad_hat, m, v

    @staticmethod
    def check_is_deterministic(
        theta_fn: Callable[..., T],
        theta_fn_kwargs: Dict[str, Any],
        synth: Callable[[T, ...], T],
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        theta_hat_1 = theta_fn(**theta_fn_kwargs)
        theta_hat_2 = theta_fn(**theta_fn_kwargs)
        if not tr.allclose(theta_hat_1, theta_hat_2):
            log.warning(
                f"theta_fn is not deterministic: "
                f"theta_hat_1 = {theta_hat_1}, theta_hat_2 = {theta_hat_2}"
            )
            return False
        if synth_fn_kwargs is None:
            synth_fn_kwargs = {}
        x_hat_1 = synth(theta_hat_1, **synth_fn_kwargs)
        x_hat_2 = synth(theta_hat_2, **synth_fn_kwargs)
        if not tr.allclose(x_hat_1, x_hat_2):
            log.warning(f"synth_fn is not deterministic")
            return False
        return True
