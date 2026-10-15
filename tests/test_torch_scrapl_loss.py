import tempfile
from collections import Counter
from pathlib import Path

import pytest

from scrapl._dependencies import require_backend

try:
    torch = require_backend("torch")
except ModuleNotFoundError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)


def test_scrapl_loss() -> None:
    # Import SCRAPLLoss from the scrapl Python package
    from scrapl.torch import SCRAPLLoss

    # Initialize SCRAPLLoss with the minimum required arguments
    scrapl_loss = SCRAPLLoss(
        shape=48000,  # Length of x and x_target in samples
        J=12,  # Number of octaves (1st and 2nd order temporal filters)
        Q1=8,  # Wavelets per octave (1st order temporal filters)
        Q2=2,  # Wavelets per octave (2nd order temporal filters)
        J_fr=3,  # Number of octaves (2nd order frequential filters)
        Q_fr=2,  # Wavelets per octave (2nd order frequential filters)
    )

    import torch as tr

    # Create two random tensors of shape (batch_size, num_channels, signal_length)
    # (x requires grad here to stand in for the output of a model being trained)
    x = tr.randn(4, 1, 48000, requires_grad=True)  # Example input tensor
    x_target = tr.randn(4, 1, 48000)  # Example target tensor
    # Compute the SCRAPL loss between x and x_target. Since SCRAPL is stochastic,
    # the loss value will be different each time `scrapl_loss()` is called.
    loss = scrapl_loss(x, x_target)
    # backward() is needed to update path_counts (and the P-Adam / P-SAGA state),
    # since they are only updated in the backward pass, not in the forward pass
    loss.backward()
    print(f"SCRAPL loss: {loss.item()}")
    first_path_idx = scrapl_loss.last_sampled_path_idx

    print(f"Number of scattering paths: {scrapl_loss.n_paths}")
    print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
    print(
        f"Most recently sampled path index (prev. example): "
        f"{scrapl_loss.last_sampled_path_idx}"
    )

    # Calculate the loss for a specific path index
    loss = scrapl_loss(x, x_target, path_idx=8)
    # backward() is needed for this path to be counted in path_counts
    loss.backward()
    print(
        f"Most recently sampled path index (specific): "
        f"{scrapl_loss.last_sampled_path_idx}"
    )
    assert scrapl_loss.last_sampled_path_idx == 8

    # Calculate the loss with a random seed for deterministic path sampling
    # (this will sample the same path index every time)
    loss = scrapl_loss(x, x_target, seed=42)
    # backward() is needed for this path to be counted in path_counts
    loss.backward()
    print(
        f"Most recently sampled path index (random seed): "
        f"{scrapl_loss.last_sampled_path_idx}"
    )
    print(f"Path sampling statistics (original): {scrapl_loss.path_counts}")
    seed_path_idx = scrapl_loss.last_sampled_path_idx
    expected_counts = Counter([first_path_idx, 8, seed_path_idx])
    assert dict(scrapl_loss.path_counts) == expected_counts

    # Get the state dictionary of the SCRAPLLoss instance
    state_dict = scrapl_loss.state_dict()

    # Clear all state of the SCRAPLLoss instance
    scrapl_loss.clear()
    print(f"Path sampling statistics (cleared): {scrapl_loss.path_counts}")
    assert not scrapl_loss.path_counts

    # Load a state dictionary into the SCRAPLLoss instance
    scrapl_loss.load_state_dict(state_dict)
    print(f"Path sampling statistics (loaded): {scrapl_loss.path_counts}")
    assert dict(scrapl_loss.path_counts) == expected_counts

    from torch import nn

    # Toy example model
    model = nn.Sequential(
        nn.Linear(in_features=48000, out_features=8),
        nn.PReLU(),
        nn.Linear(in_features=8, out_features=48000),
        nn.Tanh(),
    )

    # Attach the learnable weights of the model to the SCRAPLLoss instance
    # for P-Adam and P-SAGA
    scrapl_loss.attach_params(model.parameters())

    # Since we are using P-Adam and / or P-SAGA, we should use vanilla SGD with no
    # momentum and optional weight decay as the downstream optimizer
    from torch.optim import SGD

    sgd_optimizer = SGD(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Example forward and backward step with P-Adam and P-SAGA now active
    sgd_optimizer.zero_grad()
    x_hat = model(x)
    loss = scrapl_loss(x_hat, x_target)
    loss.backward()
    # Even though vanilla SGD is called here, the gradients of the model parameters
    # have been modified by P-Adam and P-SAGA under the hood during the backward pass
    sgd_optimizer.step()

    # To detach parameters (generally not necessary), simply attach an empty list
    scrapl_loss.attach_params([])

    print("Passed test_scrapl_loss()")


def test_scrapl_loss_warmup(tmp_path: Path) -> None:
    import torch as tr
    from torch import nn
    from scrapl import SCRAPLLoss

    # Setup
    tr.set_printoptions(precision=4, sci_mode=False)
    tr.manual_seed(42)
    bs = 3
    n_ch = 1
    n_samples = 2048
    # Number of parameters output by the encoder and input to the decoder (synth)
    n_theta = 2
    # Number of batches to use for warmup; when possible, one large batch filling all
    # available GPU memory is more efficient than many smaller batches
    n_batches = 1

    # Provide a neural network encoder that outputs `n_theta` parameters
    encoder = nn.Sequential(
        nn.Linear(n_samples, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_theta),
        nn.Sigmoid(),
    )
    # Make the encoder forward call functional (stateless)
    theta_fn = lambda x: encoder(x.squeeze(1))

    # Provide a differentiable decoder (synth) that takes as input `n_theta` parameters
    decoder = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )
    # Make the decoder forward call functional (stateless) and ensure it outputs the
    # correct shape for the SCRAPLLoss (bs, n_ch, n_samples)
    synth_fn = lambda theta: decoder(theta).unsqueeze(1)

    # Create a list of dictionaries with batches of input data for `theta_fn`;
    # typically this would be gathered from a dataloader
    theta_is_batches = [tr.rand((bs, n_ch, n_samples)) for _ in range(n_batches)]
    theta_fn_kwargs = [{"x": batch} for batch in theta_is_batches]

    # Get the trainable weights of the encoder to pass to the warmup function
    params = [p for p in encoder.parameters()]

    # Initialize SCRAPLLoss with the minimum required arguments and `n_theta`
    scrapl_loss = SCRAPLLoss(
        shape=n_samples,
        J=3,
        Q1=1,
        Q2=1,
        J_fr=1,
        Q_fr=1,
        n_theta=n_theta,
    )
    # We see that at initialization, all path sampling probabilities are uniform
    print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
    print(
        f"[min, max] path sampling probabilities (before warmup): "
        f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
    )

    # The encoder and decoder (synth) must be deterministic during warmup,
    # but can be non-deterministic otherwise
    scrapl_loss.check_is_deterministic(theta_fn, theta_fn_kwargs[0], synth_fn)

    # Run the warmup. This can be parallelized across paths by specifying different
    # `start_path_idx` and `end_path_idx` subsets on multiple devices.
    warmup_dir = str(tmp_path / "scrapl_warmup")
    scrapl_loss.warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
        save_dir=warmup_dir,
    )
    # We see that after warmup, the path sampling probabilities are no longer uniform
    print(
        f"[min, max] path sampling probabilities (after warmup): "
        f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
    )

    scrapl_loss.clear()
    print(
        f"[min, max] path sampling probabilities: "
        f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
    )

    # If warmup was conducted in parallel across multiple devices, the path sampling
    # probabilities can be loaded from a directory
    scrapl_loss.load_probs_from_warmup_dir(warmup_dir=warmup_dir)
    print(
        f"[min, max] path sampling probabilities: "
        f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
    )

    # scrapl.warmup_lc_hess_multibatch(
    #     theta_fn=theta_fn,
    #     synth_fn=synth_fn,
    #     theta_fn_kwargs=theta_fn_kwargs,
    #     params=params,
    #     n_iter=2,
    # )
    # theta_fn_kwargs = theta_fn_kwargs[0]
    # scrapl.warmup_lc_hess(
    #     theta_fn=theta_fn,
    #     synth_fn=synth_fn,
    #     theta_fn_kwargs=theta_fn_kwargs,
    #     params=params,
    #     n_iter=2,
    # )

    # INFO:__main__:theta_eigs = tensor([0.0039, 0.0011, 0.0081])
    # INFO:__main__:theta_eigs = tensor([0.0079, 0.0023, 0.0161])

    # save_path = os.path.join(OUT_DIR, "scrapl.pt")
    # tr.save(scrapl.state_dict(), save_path)
    # state_dict = tr.load(save_path)
    # scrapl.load_state_dict(state_dict)

    print("Passed test_scrapl_loss_warmup()")


if __name__ == "__main__":
    test_scrapl_loss()
