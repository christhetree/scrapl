import torch as tr
from torch import nn

from scrapl import SCRAPLLoss


def test_scrapl_loss() -> None:
    # Import SCRAPLLoss from the scrapl Python package
    from scrapl import SCRAPLLoss

    # Initialize SCRAPLLoss with the minimum required hyperparameters
    scrapl_loss = SCRAPLLoss(
        shape=48000,  # Length of x and x_target in samples
        J=12,         # Number of octaves (1st order temporal filters)
        Q1=8,         # Filters per octave (1st order temporal filters)
        Q2=2,         # Filters per octave (2nd order temporal filters)
        J_fr=3,       # Number of octaves (2nd order frequential filters)
        Q_fr=2,       # Filters per octave (2nd order frequential filters)
    )

    # Import torch and make x and x_target
    import torch as tr
    x = tr.rand((4, 1, 48000))        # Batch of 4 mono audio samples
    x_target = tr.rand((4, 1, 48000)) # Batch of 4 mono audio samples

    loss = scrapl_loss(x, x_target)
    print(f"SCRAPLLoss: {loss.item()}")

    print(f"Number of scattering paths: {scrapl_loss.n_paths}")
    print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
    print(f"Most recently sampled path index (prev. example): {scrapl_loss.curr_path_idx}")

    # Calculate the loss for a specific path index
    loss = scrapl_loss(x, x_target, path_idx=8)
    print(f"Most recently sampled path index (specific): {scrapl_loss.curr_path_idx}")

    # Calculate the loss with a random seed for deterministic path sampling
    # (this will sample the same path index every time)
    loss = scrapl_loss(x, x_target, seed=42)
    print(f"Most recently sampled path index (random seed): {scrapl_loss.curr_path_idx}")
    print(f"Path sampling statistics (original): {scrapl_loss.path_counts}")

    # Get the state dictionary of the SCRAPLLoss instance
    state_dict = scrapl_loss.state_dict()

    # Clear all state of the SCRAPLLoss instance
    scrapl_loss.clear()
    print(f"Path sampling statistics (cleared): {scrapl_loss.path_counts}")

    # Load a state dictionary into the SCRAPLLoss instance
    scrapl_loss.load_state_dict(state_dict)
    print(f"Path sampling statistics (loaded): {scrapl_loss.path_counts}")


def test_scrapl_loss_warmup() -> None:
    # Setup
    tr.set_printoptions(precision=4, sci_mode=False)
    tr.manual_seed(0)
    bs = 2
    n_samples = 32768
    n_theta = 3

    model = nn.Sequential(
        nn.Linear(n_samples, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_theta),
        nn.Sigmoid(),
    )
    model = model
    synth = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )
    synth = synth
    loss_fn = nn.MSELoss()
    x_1 = tr.rand((bs, n_samples))
    x_2 = tr.rand((bs, n_samples))
    x_3 = tr.rand((bs, n_samples))

    params = [p for p in model.parameters()]
    assert all(not p.grad for p in params)

    scrapl = SCRAPLLoss(
        shape=n_samples,
        J=12,
        Q1=8,
        Q2=2,
        J_fr=3,
        Q_fr=2,
        n_theta=n_theta,
        sample_all_paths_first=False,
    )
    # TODO: Check whether probs are loaded in ckpt or need to be in buffer
    # scrapl.attach_params(params)
    theta_fn = lambda x: model(x.squeeze(1))
    synth_fn = lambda theta: synth(theta).unsqueeze(1)

    theta_fn_kwargs = [
        {"x": x_1.unsqueeze(1)},
        # {"x": x_1.unsqueeze(1)},
        # {"x": x_2.unsqueeze(1)},
        # {"x": x_3.unsqueeze(1)},
    ]
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
    scrapl.warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
        n_iter=20,
        agg="none",
        # agg="mean",
        force_multibatch=False,
        # force_multibatch=True,
    )

    # INFO:__main__:theta_eigs = tensor([0.0039, 0.0011, 0.0081])
    # INFO:__main__:theta_eigs = tensor([0.0079, 0.0023, 0.0161])

    # save_path = os.path.join(OUT_DIR, "scrapl.pt")
    # tr.save(scrapl.state_dict(), save_path)
    # state_dict = tr.load(save_path)
    # scrapl.load_state_dict(state_dict)


if __name__ == "__main__":
    test_scrapl_loss()

    # test_scrapl_loss()
