import torch

from downscale.precipitation import detail_band_crps


def test_detail_crps_detects_structure_hidden_by_block_average():
    truth = torch.tensor([[[0., 2.], [2., 0.]]])
    flat = torch.ones(1, 2, 2, 2, requires_grad=True)
    score = detail_band_crps(flat, truth)
    torch.testing.assert_close(score, torch.tensor(1.))
    score.backward()
    assert flat.grad.abs().sum() > 0
    identical = truth[:, None].expand(-1, 2, -1, -1)
    torch.testing.assert_close(detail_band_crps(identical, truth), torch.tensor(0.))


def test_multiscale_loss_on_odd_grid_has_finite_gradients():
    pred = torch.rand(2, 4, 9, 11, requires_grad=True)
    truth = torch.rand(2, 9, 11)
    detail_band_crps(pred, truth).backward()
    assert torch.isfinite(pred.grad).all()
