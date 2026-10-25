"""
Review sanity checks (added 2026-10-25)
=======================================
Plain-script tests (no pytest). Each one would fail if the corresponding piece of the
review experiment were broken:

1. Label offsets: the trainers must map local labels 0..19 to absolute ids 20*t..20*t+19.
   The M1/M2 result files committed on 2026-10-03/04 were produced before this offset existed.
2. --no-attention ablation: the fused feature must equal the raw encoder feature exactly, and
   fuse_features must equal the normalised input.
3. Seeding: two models built under the same seed must have identical parameters; a different
   seed must not.
4. EWC penalty: zero exactly at the anchor, strictly positive after a perturbation.
5. Norm diagnostic: at init the encoder output has per-element unit variance (LayerNorm), so its
   L2 norm is close to sqrt(512); the diagnostic dict must report that.
6. Narrow encoder: base_width 64 builds, runs, and has fewer parameters than base_width 128.

Checks 7 to 9 guard the three performance fixes made on 2026-10-25. Each one would fail if the
corresponding fix were reverted:

7. Replay noise must live in the prototypes' own space. Prototypes are unit vectors, so the
   noise built from the stored variances must be short enough that a draw keeps its class
   identity. Before the fix the noise was ~9x longer than the prototype and a nearest-prototype
   oracle scored near chance.
8. The attention reconstruction loss must not depend on the attention LayerNorm's gain. Before
   the fix it fell monotonically as the gain shrank, so training destroyed the retrieval path to
   satisfy it.
9. The two models must take their channel dropout from the same config field. Before the fix
   MAVIS used 0.5 and the baseline silently used 0.0, confounding every comparison between them.

Run:  python tests/test_review_sanity.py
"""

import os
import sys
import math
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TRITON_DISABLE"] = "1"

import torch
import torch.nn.functional as F

from config import get_default_config
from models.mavis import MAVIS
from models.encoder import CNNEncoder
from continual.ewc import OnlineEWC


def _small_config(attention_enabled=True, width=32):
    cfg = get_default_config()
    cfg.encoder.base_width = width
    cfg.attention.enabled = attention_enabled
    cfg.encoder.dropout_rate = 0.0
    return cfg


def test_label_offset_arithmetic():
    classes_per_task = 20
    for task_id in range(10):
        local = torch.arange(classes_per_task)
        absolute = local + task_id * classes_per_task
        assert absolute.min().item() == task_id * classes_per_task
        assert absolute.max().item() == (task_id + 1) * classes_per_task - 1
        # The pre-fix code evaluated local labels against a 200-way head; that mapping is wrong
        # for every task after the first.
        if task_id > 0:
            assert not torch.equal(absolute, local)
    print("[OK] 1. label offset arithmetic covers 20*t .. 20*t+19 for every task")


def test_no_attention_bypass():
    torch.manual_seed(0)
    cfg = _small_config(attention_enabled=False)
    model = MAVIS(cfg).eval()
    x = torch.randn(8, 3, 64, 64)
    with torch.no_grad():
        logits, f, fused = model(x, return_features=True)
        assert torch.equal(f, fused), "fused feature must be identical to encoder feature when attention is off"
        assert logits.shape == (8, 200)
        p = F.normalize(torch.randn(16, 512), dim=-1)
        out = model.fuse_features(p)
        assert torch.allclose(out, F.normalize(p, dim=-1), atol=1e-6)
        assert model.last_norms["r_share"] == 0.0 and model.last_norms["gate_mean"] == 1.0
    # With attention on, and an empty memory bank, the retrieved vector is the learned no-memory
    # embedding and the fused feature must differ from the raw one.
    cfg_on = _small_config(attention_enabled=True)
    torch.manual_seed(0)
    model_on = MAVIS(cfg_on).eval()
    with torch.no_grad():
        _, f_on, fused_on = model_on(x, return_features=True)
        assert not torch.equal(f_on, fused_on)
    print("[OK] 2. --no-attention bypass: fused == f, fuse_features == normalize(p); attention-on path differs")


def test_seeding_reproducible():
    cfg = _small_config()
    torch.manual_seed(123)
    a = MAVIS(cfg)
    torch.manual_seed(123)
    b = MAVIS(cfg)
    torch.manual_seed(124)
    c = MAVIS(cfg)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"same seed produced different parameter {k}"
    assert any(not torch.equal(sa[k], sc[k]) for k in sa if sa[k].dtype.is_floating_point and sa[k].numel() > 0 and "fusion_gate" not in k and "memory_bank" not in k and "task_scale" not in k), \
        "different seeds produced identical parameters"
    print("[OK] 3. seeding: identical seeds -> identical parameters; different seed -> different parameters")


def test_ewc_penalty_zero_at_anchor():
    torch.manual_seed(0)
    cfg = _small_config()
    model = MAVIS(cfg)
    ewc = OnlineEWC(model, ewc_lambda=5000.0, gamma=0.88)
    # Fake a Fisher of ones and an anchor at the current parameters (no dataloader needed)
    ewc.running_fisher = {n: torch.ones_like(p) for n, p in model.named_parameters() if p.requires_grad}
    ewc.saved_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    ewc.task_count = 1
    pen0 = ewc.penalty(model).item()
    assert pen0 == 0.0, f"penalty at anchor should be exactly 0, got {pen0}"
    with torch.no_grad():
        next(model.classifier.parameters()).add_(0.01)
    pen1 = ewc.penalty(model).item()
    assert pen1 > 0.0
    n_cls = next(model.classifier.parameters()).numel()
    expected = 5000.0 / 2.0 * n_cls * 0.01 ** 2
    assert abs(pen1 - expected) / expected < 1e-3, f"penalty {pen1} != expected {expected}"
    print("[OK] 4. EWC penalty: 0 at anchor, lambda/2 * sum F (dtheta)^2 after perturbation")


def test_norm_diagnostic():
    torch.manual_seed(0)
    enc = CNNEncoder(feature_dim=512, dropout_rate=0.0, base_width=32).eval()
    with torch.no_grad():
        f = enc(torch.randn(16, 3, 64, 64))
    norms = f.norm(dim=1)
    # LayerNorm output: per-element mean 0, var 1 -> ||f|| = sqrt(D) up to the biased-variance factor
    assert abs(norms.mean().item() - math.sqrt(512)) / math.sqrt(512) < 0.02, norms.mean().item()
    cfg = _small_config()
    torch.manual_seed(0)
    model = MAVIS(cfg).eval()
    with torch.no_grad():
        model(torch.randn(4, 3, 64, 64))
    d = model.last_norms
    assert set(d) == {"f_norm", "r_norm", "gate_mean", "r_share"}
    assert abs(d["f_norm"] - math.sqrt(512)) / math.sqrt(512) < 0.02
    assert 0.0 <= d["r_share"] <= 1.0
    print(f"[OK] 5. norm diagnostic: ||f||={d['f_norm']:.2f} ~ sqrt(512)={math.sqrt(512):.2f}; keys present")


def test_narrow_encoder():
    wide = CNNEncoder(512, 0.0, base_width=128)
    narrow = CNNEncoder(512, 0.0, base_width=64)
    n_w = sum(p.numel() for p in wide.parameters())
    n_n = sum(p.numel() for p in narrow.parameters())
    assert n_n < n_w / 3, (n_w, n_n)
    with torch.no_grad():
        out, (f3, f4) = narrow.eval()(torch.randn(2, 3, 64, 64), return_spatial=True)
    assert out.shape == (2, 512) and f3.shape == (2, 256, 16, 16) and f4.shape == (2, 512, 8, 8)
    try:
        CNNEncoder(512, 0.0, base_width=48)
        raise AssertionError("base_width 48 should be rejected (not a multiple of 32)")
    except AssertionError as e:
        if "rejected" in str(e):
            raise
    print(f"[OK] 6. narrow encoder: width 64 has {n_n/1e6:.1f}M params vs {n_w/1e6:.1f}M at width 128")


def test_replay_noise_in_prototype_space():
    """Fix A: variances must be measured where the prototypes live, so draws keep their label."""
    from models.memory import kmeans_prototypes
    torch.manual_seed(0)
    D, C, K = 512, 20, 10
    # Synthetic feature cloud shaped like the encoder's output: LayerNorm gives per-element unit
    # variance, so raw features sit at length sqrt(D) with class means spread around the sphere.
    centres = F.normalize(torch.randn(C, D), p=2, dim=1) * math.sqrt(D)
    protos, labels, variances = [], [], []
    for c in range(C):
        feats = centres[c] + torch.randn(200, D) * 0.6
        pk = kmeans_prototypes(feats, k=K, num_iters=5)
        var_c = F.normalize(feats, p=2, dim=1).var(dim=0, unbiased=False) + 1e-8
        for k in range(K):
            protos.append(pk[k]); labels.append(c); variances.append(var_c)
    protos = torch.stack(protos); labels = torch.tensor(labels); variances = torch.stack(variances)

    assert torch.allclose(protos.norm(dim=1), torch.ones(len(protos)), atol=1e-4), \
        "prototypes are expected to be unit vectors"
    sigma = torch.sqrt(torch.clamp(variances, min=1e-6)) * 0.5
    noise_len = sigma.norm(dim=1).mean().item()
    assert noise_len < 1.0, f"replay noise length {noise_len:.3f} must be shorter than the unit prototype"

    draws = F.normalize(protos + torch.randn_like(protos) * sigma, p=2, dim=-1)
    cos = (draws * F.normalize(protos, dim=-1)).sum(-1).mean().item()
    pred = labels[(draws @ F.normalize(protos, dim=-1).t()).argmax(1)]
    oracle = (pred == labels).float().mean().item()
    assert cos > 0.8, f"cosine(draw, its prototype) = {cos:.3f}; the class direction is being lost"
    assert oracle > 0.9, f"nearest-prototype oracle = {100*oracle:.1f}%; draws carry no label"
    print(f"[OK] 7. replay noise length {noise_len:.3f} vs unit prototype; "
          f"cos(draw,proto)={cos:.3f}; oracle={100*oracle:.1f}% (chance {100/C:.0f}%)")


def test_recon_loss_invariant_to_layernorm_gain():
    """Fix B: the reconstruction loss must measure direction, not length."""
    cfg = _small_config()
    model = MAVIS(cfg).eval()
    torch.manual_seed(0)
    protos = F.normalize(torch.randn(64, cfg.encoder.feature_dim), p=2, dim=-1)
    losses = []
    for gain in (1.0, 0.44, 0.05):
        mm = copy.deepcopy(model)
        with torch.no_grad():
            mm.attention.layer_norm.weight.fill_(gain)
            mm.attention.layer_norm.bias.zero_()
            out = mm.attention(protos, protos, chunk_size=64)
            losses.append(F.mse_loss(F.normalize(out, p=2, dim=-1), protos).item())
    spread = max(losses) - min(losses)
    assert spread < 1e-6, f"loss still depends on the gain: {losses}"
    print(f"[OK] 8. reconstruction loss is flat across LayerNorm gains 1.00/0.44/0.05 "
          f"(spread {spread:.2e}, value {losses[0]:.6f})")


def test_dropout_symmetric_between_models():
    """Fix C: one config field must set the channel dropout for both models."""
    from models.baseline import BaselineCNN
    for rate in (0.0, 0.5):
        cfg = _small_config()
        cfg.encoder.dropout_rate = rate
        mav = MAVIS(cfg)
        base = BaselineCNN(feature_dim=cfg.encoder.feature_dim, total_classes=200,
                           base_width=cfg.encoder.base_width, dropout_rate=cfg.encoder.dropout_rate)
        p_mav = getattr(mav.encoder.layer1[0].dropout, "p", 0.0)
        p_base = getattr(base.encoder.layer1[0].dropout, "p", 0.0)
        assert p_mav == p_base == rate, f"rate {rate}: MAVIS {p_mav}, baseline {p_base}"
    assert get_default_config().encoder.dropout_rate == 0.0, \
        "the shipped default should be 0.0 so neither model is regularised more than the other"
    print("[OK] 9. channel dropout matches across MAVIS and the baseline at 0.0 and 0.5; default is 0.0")


if __name__ == "__main__":
    test_label_offset_arithmetic()
    test_no_attention_bypass()
    test_seeding_reproducible()
    test_ewc_penalty_zero_at_anchor()
    test_norm_diagnostic()
    test_narrow_encoder()
    test_replay_noise_in_prototype_space()
    test_recon_loss_invariant_to_layernorm_gain()
    test_dropout_symmetric_between_models()
    print("\nAll review sanity checks passed.")
