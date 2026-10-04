"""C6 计时复测（同一台 Apple M5 Pro）：表S8 / 正文§5.4 的单窗推理耗时；表S60 / 正文§5.4 的每步训练耗时（MPS，batch 256）。
训练步与 tools/run_e9_train.py 的循环逐项相同（抽批、搬运、前向、反向、裁剪、AdamW、OneCycle，确定性算法 warn_only）。
一次只占用一个 MPS 进程。计时本身随系统状态波动，这里只判断量级与相对快慢。"""
import sys, time, random, json
import numpy as np
import torch
from pathlib import Path
P = str(Path(__file__).resolve().parents[1])
sys.path[:0] = [P + '/src', P + '/tools']
import pkg_train as T, sensors_frontend as SF, e9_eqnio_frontend as E9
from sensors_evaluate import load

torch.set_num_threads(4)
mps = torch.device('mps')


def sync(dev):
    if dev.type == 'mps':
        torch.mps.synchronize()


# ---------------------------------------------------------------- 推理（batch 1，30 预热，100 计时）
src = json.load(open(P + '/results/sensors_v4/diagnostics.json'))
w = src['rotation_window_sources'][0]
x1 = torch.from_numpy(np.asarray(np.load(P + '/' + w['file'])['feat'][w['start']:w['start'] + 200].T, np.float32))[None]
ref = {(t['model'], t['device']): (t['median_ms'], t['p95_ms']) for t in src['timings']}
for dev in (torch.device('cpu'), mps):
    for tag in ('ResNet18_v3_hn_gn_s0', 'Sensors_v4_resnet_gn_s0', 'A6_v3_nopre_s0', 'IMUNet2024_v3_hn_gn_s0'):
        net, _, _, _ = load(tag, dev); xs = x1.to(dev); el = []
        with torch.inference_mode():
            for _ in range(30):
                net(xs)
            sync(dev)
            for _ in range(100):
                t0 = time.perf_counter(); net(xs); sync(dev); el.append(1000 * (time.perf_counter() - t0))
        med, p95 = np.median(el), np.quantile(el, .95); rm, rp = ref[(tag, dev.type)]
        print(f'infer.{dev.type}.{tag} = median {med:.3f} ms (archived {rm:.3f}), p95 {p95:.3f} (archived {rp:.3f}), ratio {med / rm:.2f}')

# ---------------------------------------------------------------- 训练步（MPS，batch 256）
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
train_data, window, _ = T.load_split('train', ('ronin',))
mu, sd = T.global_norm_stats(train_data)
cfgs = {'ResNet18+GN+HN': lambda: SF.build('heading', mu, sd, 'ronin_resnet18', False, 1.0),
        'EqNIO SO(2)': lambda: E9.build('eqnio_so2', mu, sd), 'EqNIO O(2)': lambda: E9.build('eqnio_o2', mu, sd)}
out = {}
for name, mk in cfgs.items():
    net = mk().to(mps); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=20000, pct_start=.15)
    loss_fn = T.make_loss('huber', .27); rng = np.random.default_rng(0)

    def step():
        x, y = T.sample_batch(train_data, 256, rng, window)
        x, y = x.to(mps), y.to(mps)
        loss = loss_fn(net(x), y); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step(); sch.step()
    for _ in range(20):
        step()
    sync(mps); t0 = time.perf_counter(); n = 200
    for _ in range(n):
        step()
    sync(mps); out[name] = (time.perf_counter() - t0) / n
    print(f'train_step.{name} = {out[name]:.4f} s/step ({sum(p.numel() for p in net.parameters())} params)', flush=True)
    del net, opt; torch.mps.empty_cache()
print(f'train_step.ratio_so2_over_resnet = {out["EqNIO SO(2)"] / out["ResNet18+GN+HN"]:.1f}')
print(f'train_step.ratio_o2_over_resnet = {out["EqNIO O(2)"] / out["ResNet18+GN+HN"]:.1f}')
print(f'train_step.o2_faster_than_so2 = {out["EqNIO O(2)"] < out["EqNIO SO(2)"]}')
