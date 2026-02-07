import torch as tr
from torch import nn

from scrapl import SCRAPLLoss


def test_scrapl_loss() -> None:
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
