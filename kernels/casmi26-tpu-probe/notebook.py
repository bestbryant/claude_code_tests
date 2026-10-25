"""TPU feasibility probe: which frameworks the Kaggle TPU VM image ships and how fast a PeakFormer-sized
transformer step runs on one XLA core (static shapes, bf16)."""
import subprocess, sys, time
print(sys.version, flush=True)
subprocess.run('pip list 2>/dev/null | grep -i -E "^(torch|jax|jaxlib|flax|optax|libtpu|tensorflow|numpy) "', shell=True)
try:
    import torch, torch_xla, torch_xla.core.xla_model as xm
    print('torch', torch.__version__, 'torch_xla', torch_xla.__version__, flush=True)
    try:
        import torch_xla.runtime as xr
        print('devices', xr.global_runtime_device_count(), flush=True)
    except Exception as e:
        print('runtime info failed', repr(e))
    dev = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
    layer = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(384, 8, 1536, 0.0, batch_first=True), 6).to(dev)
    head = torch.nn.Linear(384, 15440).to(dev)
    opt = torch.optim.AdamW(list(layer.parameters()) + list(head.parameters()), 1e-4)
    for B in (512, 1024):
        x = torch.randn(B, 128, 384, device=dev); y = (torch.rand(B, 15440, device=dev) < 0.02).float()
        for i in range(12):
            if i == 2:
                xm.mark_step(); xm.wait_device_ops(); t = time.time()
            with torch.autocast('xla', dtype=torch.bfloat16):
                h = layer(x)[:, 0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(h).float(), y)
            opt.zero_grad(); loss.backward(); opt.step(); xm.mark_step()
        xm.wait_device_ops()
        print(f'torch_xla 1 core: d384 x 6, L128, batch {B}: {(time.time() - t) / 10:.3f} s/step, loss {loss.item():.4f}', flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
try:
    import jax
    print('jax', jax.__version__, jax.devices(), flush=True)
except Exception as e:
    print('jax failed', repr(e))
