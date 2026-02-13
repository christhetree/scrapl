import shutil

import torch as tr
from torch import nn

from scrapl import SCRAPLLoss


def test_scrapl_loss() -> None:
    # Import SCRAPLLoss from the scrapl Python package
    from scrapl import SCRAPLLoss

    # Initialize SCRAPLLoss with the minimum required arguments
    scrapl_loss = SCRAPLLoss(
        shape=48000,  # Length of x and x_target in samples
        J=12,  # Number of octaves (1st order temporal filters)
        Q1=8,  # Filters per octave (1st order temporal filters)
        Q2=2,  # Filters per octave (2nd order temporal filters)
        J_fr=3,  # Number of octaves (2nd order frequential filters)
        Q_fr=2,  # Filters per octave (2nd order frequential filters)
    )

    # Import torch and make x and x_target
    import torch as tr

    x = tr.rand((4, 1, 48000))  # Batch of 4 mono audio samples
    x_target = tr.rand((4, 1, 48000))  # Batch of 4 mono audio samples

    loss = scrapl_loss(x, x_target)
    print(f"SCRAPLLoss: {loss.item()}")

    print(f"Number of scattering paths: {scrapl_loss.n_paths}")
    print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
    print(
        f"Most recently sampled path index (prev. example): {scrapl_loss.curr_path_idx}"
    )

    # Calculate the loss for a specific path index
    loss = scrapl_loss(x, x_target, path_idx=8)
    print(f"Most recently sampled path index (specific): {scrapl_loss.curr_path_idx}")

    # Calculate the loss with a random seed for deterministic path sampling
    # (this will sample the same path index every time)
    loss = scrapl_loss(x, x_target, seed=42)
    print(
        f"Most recently sampled path index (random seed): {scrapl_loss.curr_path_idx}"
    )
    print(f"Path sampling statistics (original): {scrapl_loss.path_counts}")

    # Get the state dictionary of the SCRAPLLoss instance
    state_dict = scrapl_loss.state_dict()

    # Clear all state of the SCRAPLLoss instance
    scrapl_loss.clear()
    print(f"Path sampling statistics (cleared): {scrapl_loss.path_counts}")

    # Load a state dictionary into the SCRAPLLoss instance
    scrapl_loss.load_state_dict(state_dict)
    print(f"Path sampling statistics (loaded): {scrapl_loss.path_counts}")

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


def test_scrapl_loss_warmup() -> None:
    import torch as tr
    from scrapl import SCRAPLLoss

    # Setup
    tr.set_printoptions(precision=4, sci_mode=False)
    tr.manual_seed(42)
    bs = 4
    n_ch = 1
    n_samples = 8096
    # Number of parameters output by the encoder and input to the decoder (synth)
    n_theta = 3
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
        J_fr=2,
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
    scrapl_loss.warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
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
    scrapl_loss.load_probs_from_warmup_dir(warmup_dir="scrapl_warmup")
    print(
        f"[min, max] path sampling probabilities: "
        f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
    )

    # Remove the warmup directory after loading the probabilities
    shutil.rmtree("scrapl_warmup")

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


if __name__ == "__main__":
    test_scrapl_loss()
    test_scrapl_loss_warmup()
