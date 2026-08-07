"""
XPINN Stokes-Darcy - CAS 1
Domaines rectangulaires, solution continue à l'interface (y = 0).
"""

import os, math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    nu: float = 0.1
    K: float = 0.01
    alpha: float = 1.0

    Nus: int = 500
    Nud: int = 500
    NuGamma: int = 125
    Nfs: int = 8000
    Nfd: int = 8000

    hidden_width: int = 100
    hidden_depth: int = 5

    epochs_adam: int = 7000
    epochs_lbfgs: int = 320
    use_lbfgs: bool = True
    adam_lr: float = 1e-3

    use_weighting: bool = True
    w_gauge: float = 1.0e-4

    output_dir: str = "outputs_case1"
    seed: int = 1234
    dpi: int = 250
    eval_grid: int = 200
    dtype: torch.dtype = torch.float64

def update_weights(cfg):
    if cfg.use_weighting:
        cfg.weight_stokes = 1.0 / cfg.nu
        cfg.weight_darcy = cfg.K / cfg.nu
    else:
        cfg.weight_stokes = 1.0
        cfg.weight_darcy = 1.0

CFG = Config()
update_weights(CFG)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.pi = math.pi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_default_dtype(CFG.dtype)
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

# ============================================================
# UTILITAIRES
# ============================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        retain_graph=True, create_graph=True
    )[0]

def lhs_1d(n, a, b, device_, dtype_):
    u = (torch.arange(n, device=device_, dtype=dtype_) + torch.rand(n, device=device_, dtype=dtype_)) / n
    return a + (b - a) * u[torch.randperm(n)]

def lhs_2d(n, ax, bx, ay, by, device_, dtype_):
    return lhs_1d(n, ax, bx, device_, dtype_).reshape(-1, 1), lhs_1d(n, ay, by, device_, dtype_).reshape(-1, 1)

def split_counts(total, parts):
    base = total // parts
    rem = total % parts
    out = [base] * parts
    for i in range(rem):
        out[i] += 1
    return tuple(out)

# ============================================================
# SOLUTION EXACTE - CAS 1
# ============================================================

class ExactSolutionCase1:
    """Solution continue à l'interface, domaines rectangulaires"""
    @staticmethod
    def u(x, y):
        return -torch.sin(math.pi * x)**2 * torch.sin(math.pi * y) * torch.cos(math.pi * y)
    @staticmethod
    def v(x, y):
        return torch.sin(math.pi * x) * torch.cos(math.pi * x) * torch.sin(math.pi * y)**2
    @staticmethod
    def p(x, y):
        return torch.sin(math.pi * x) * torch.cos(math.pi * y)

# ============================================================
# RÉSEAUX DE NEURONES
# ============================================================

class StokesNet(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inp = torch.cat((x, y), dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

class DarcyNet(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inp = torch.cat((x, y), dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

# ============================================================
# DONNÉES D'ENTRAÎNEMENT
# ============================================================

@dataclass
class TrainingData:
    xs_f: torch.Tensor; ys_f: torch.Tensor
    xd_f: torch.Tensor; yd_f: torch.Tensor
    xs_b: torch.Tensor; ys_b: torch.Tensor
    xd_b: torch.Tensor; yd_b: torch.Tensor
    x_g: torch.Tensor; y_gs: torch.Tensor; y_gd: torch.Tensor
    f_stokes_1: torch.Tensor; f_stokes_2: torch.Tensor
    f_darcy_1: torch.Tensor; f_darcy_2: torch.Tensor
    ub_s: torch.Tensor; vb_s: torch.Tensor
    ub_d: torch.Tensor; vb_d: torch.Tensor
    g1: torch.Tensor; g2: torch.Tensor

def build_training_data(cfg):
    dtype_ = cfg.dtype
    Exact = ExactSolutionCase1

    xs_f, ys_f = lhs_2d(cfg.Nfs, 0., 1., 0., 1., device, dtype_)
    xd_f, yd_f = lhs_2d(cfg.Nfd, 0., 1., -1., 0., device, dtype_)
    xs_f = xs_f.requires_grad_(True); ys_f = ys_f.requires_grad_(True)
    xd_f = xd_f.requires_grad_(True); yd_f = yd_f.requires_grad_(True)

    x_g = lhs_1d(cfg.NuGamma, 0., 1., device, dtype_).reshape(-1, 1).requires_grad_(True)
    y_gs = torch.zeros_like(x_g).requires_grad_(True)
    y_gd = torch.zeros_like(x_g).requires_grad_(True)

    nL, nT, nR = split_counts(cfg.Nus, 3)
    yL = lhs_1d(nL, 0., 1., device, dtype_).reshape(-1, 1)
    xL = torch.zeros_like(yL)
    xT = lhs_1d(nT, 0., 1., device, dtype_).reshape(-1, 1)
    yT = torch.ones_like(xT)
    yR = lhs_1d(nR, 0., 1., device, dtype_).reshape(-1, 1)
    xR = torch.ones_like(yR)
    xs_b = torch.cat((xL, xT, xR), 0).requires_grad_(True)
    ys_b = torch.cat((yL, yT, yR), 0).requires_grad_(True)

    nL, nB, nR = split_counts(cfg.Nud, 3)
    yL = lhs_1d(nL, -1., 0., device, dtype_).reshape(-1, 1)
    xL = torch.zeros_like(yL)
    xB = lhs_1d(nB, 0., 1., device, dtype_).reshape(-1, 1)
    yB = -torch.ones_like(xB)
    yR = lhs_1d(nR, -1., 0., device, dtype_).reshape(-1, 1)
    xR = torch.ones_like(yR)
    xd_b = torch.cat((xL, xB, xR), 0).requires_grad_(True)
    yd_b = torch.cat((yL, yB, yR), 0).requires_grad_(True)

    ub_s = Exact.u(xs_b, ys_b).detach()
    vb_s = Exact.v(xs_b, ys_b).detach()
    ub_d = Exact.u(xd_b, yd_b).detach()
    vb_d = Exact.v(xd_b, yd_b).detach()

    u_e = Exact.u(xs_f, ys_f); v_e = Exact.v(xs_f, ys_f); p_e = Exact.p(xs_f, ys_f)
    u_x = grad(u_e, xs_f); u_y = grad(u_e, ys_f)
    v_x = grad(v_e, xs_f); v_y = grad(v_e, ys_f)
    p_x = grad(p_e, xs_f); p_y = grad(p_e, ys_f)
    u_xx = grad(u_x, xs_f); u_yy = grad(u_y, ys_f); u_xy = grad(u_x, ys_f)
    v_xx = grad(v_x, xs_f); v_yy = grad(v_y, ys_f); v_xy = grad(v_x, ys_f)
    f_stokes_1 = (2 * cfg.nu * u_xx - p_x + cfg.nu * (u_yy + v_xy)).detach()
    f_stokes_2 = (cfg.nu * (v_xx + u_xy) - p_y + 2 * cfg.nu * v_yy).detach()

    u_e = Exact.u(xd_f, yd_f); v_e = Exact.v(xd_f, yd_f); p_e = Exact.p(xd_f, yd_f)
    p_x = grad(p_e, xd_f); p_y = grad(p_e, yd_f)
    f_darcy_1 = (cfg.nu * (1. / cfg.K) * u_e + p_x).detach()
    f_darcy_2 = (cfg.nu * (1. / cfg.K) * v_e + p_y).detach()

    u_g = Exact.u(x_g, y_gs); v_g = Exact.v(x_g, y_gs)
    p_sg = Exact.p(x_g, y_gs); p_dg = Exact.p(x_g, y_gd)
    v_gx = grad(v_g, x_g); v_gy = grad(v_g, y_gs); u_gy = grad(u_g, y_gs)
    g1 = (2 * cfg.nu * v_gy - p_sg + p_dg).detach()
    g2 = (-(v_gx + u_gy) + cfg.alpha * (1. / cfg.K) ** 0.5 * u_g).detach()

    return TrainingData(xs_f, ys_f, xd_f, yd_f, xs_b, ys_b, xd_b, yd_b,
                        x_g, y_gs, y_gd, f_stokes_1, f_stokes_2, f_darcy_1, f_darcy_2,
                        ub_s, vb_s, ub_d, vb_d, g1, g2)

# ============================================================
# SOLVEUR XPINN
# ============================================================

class XPINNSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_s = StokesNet(cfg.hidden_width, cfg.hidden_depth).to(device)
        self.model_d = DarcyNet(cfg.hidden_width, cfg.hidden_depth).to(device)

        self.opt_s = torch.optim.Adam(self.model_s.parameters(), lr=cfg.adam_lr)
        self.opt_d = torch.optim.Adam(self.model_d.parameters(), lr=cfg.adam_lr)

        self.scheduler_s = torch.optim.lr_scheduler.StepLR(self.opt_s, step_size=2000, gamma=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.opt_d, step_size=2000, gamma=0.5)

        self.lbfgs_s = torch.optim.LBFGS(self.model_s.parameters(), max_iter=20, tolerance_grad=1e-7,
                                         tolerance_change=1e-9, history_size=100, line_search_fn="strong_wolfe")
        self.lbfgs_d = torch.optim.LBFGS(self.model_d.parameters(), max_iter=20, tolerance_grad=1e-7,
                                         tolerance_change=1e-9, history_size=100, line_search_fn="strong_wolfe")

        self.data = build_training_data(cfg)
        self.Exact = ExactSolutionCase1

        self.w_int_s = cfg.weight_stokes
        self.w_int_d = cfg.weight_darcy
        self.history = {"epoch": [], "loss_s": [], "loss_d": [],
                        "L2_u_s": [], "L2_p_s": [], "L2_u_d": [], "L2_p_d": [], "L2_sum": []}

    def loss_stokes(self):
        data = self.data
        cfg = self.cfg
        u_f, v_f, p_f = self.model_s(data.xs_f, data.ys_f)

        u_x = grad(u_f, data.xs_f); u_y = grad(u_f, data.ys_f)
        v_x = grad(v_f, data.xs_f); v_y = grad(v_f, data.ys_f)
        p_x = grad(p_f, data.xs_f); p_y = grad(p_f, data.ys_f)
        u_xx = grad(u_x, data.xs_f); u_yy = grad(u_y, data.ys_f); u_xy = grad(u_x, data.ys_f)
        v_xx = grad(v_x, data.xs_f); v_yy = grad(v_y, data.ys_f); v_xy = grad(v_x, data.ys_f)

        r1 = 2 * cfg.nu * u_xx - p_x + cfg.nu * (u_yy + v_xy) - data.f_stokes_1
        r2 = cfg.nu * (v_xx + u_xy) - p_y + 2 * cfg.nu * v_yy - data.f_stokes_2
        div = u_x + v_y
        l_int = torch.mean(r1 ** 2) + torch.mean(r2 ** 2) + torch.mean(div ** 2)

        u_b, v_b, _ = self.model_s(data.xs_b, data.ys_b)
        l_bc = torch.mean((u_b - data.ub_s) ** 2) + torch.mean((v_b - data.vb_s) ** 2)

        x_g_d = data.x_g.clone().detach().requires_grad_(True)
        y_gd_d = data.y_gd.clone().detach().requires_grad_(True)
        u_d, v_d, p_d = self.model_d(x_g_d, y_gd_d)

        u_s, v_s, p_s = self.model_s(data.x_g, data.y_gs)
        v_s_x = grad(v_s, data.x_g); v_s_y = grad(v_s, data.y_gs); u_s_y = grad(u_s, data.y_gs)

        cond1 = v_s - v_d
        cond2 = 2 * cfg.nu * v_s_y - p_s + p_d - data.g1
        cond3 = -(v_s_x + u_s_y) + cfg.alpha * (1. / cfg.K) ** 0.5 * u_s - data.g2
        l_interface = torch.mean(cond1 ** 2) + torch.mean(cond2 ** 2) + torch.mean(cond3 ** 2)

        l_gauge = torch.mean(p_f) ** 2
        return self.w_int_s * l_int + l_bc + l_interface + cfg.w_gauge * l_gauge

    def loss_darcy(self):
        data = self.data
        cfg = self.cfg
        u_f, v_f, p_f = self.model_d(data.xd_f, data.yd_f)

        u_x = grad(u_f, data.xd_f); v_y = grad(v_f, data.yd_f)
        p_x = grad(p_f, data.xd_f); p_y = grad(p_f, data.yd_f)

        r1 = cfg.nu * (1. / cfg.K) * u_f + p_x - data.f_darcy_1
        r2 = cfg.nu * (1. / cfg.K) * v_f + p_y - data.f_darcy_2
        div = u_x + v_y
        l_int = torch.mean(r1 ** 2) + torch.mean(r2 ** 2) + torch.mean(div ** 2)

        u_b, v_b, _ = self.model_d(data.xd_b, data.yd_b)
        l_bc = torch.mean((u_b - data.ub_d) ** 2) + torch.mean((v_b - data.vb_d) ** 2)

        x_g_s = data.x_g.clone().detach().requires_grad_(True)
        y_gs_s = data.y_gs.clone().detach().requires_grad_(True)
        u_s, v_s, p_s = self.model_s(x_g_s, y_gs_s)

        u_d, v_d, p_d = self.model_d(data.x_g, data.y_gd)

        v_s_x = grad(v_s, x_g_s)
        v_s_y = grad(v_s, y_gs_s)
        u_s_y = grad(u_s, y_gs_s)

        cond1 = v_s - v_d
        cond2 = 2 * cfg.nu * v_s_y - p_s + p_d - data.g1
        cond3 = -(v_s_x + u_s_y) + cfg.alpha * (1. / cfg.K) ** 0.5 * u_s - data.g2
        l_interface = torch.mean(cond1 ** 2) + torch.mean(cond2 ** 2) + torch.mean(cond3 ** 2)

        l_gauge = torch.mean(p_f) ** 2
        return self.w_int_d * l_int + l_bc + l_interface + cfg.w_gauge * l_gauge

    def compute_metrics(self):
        cfg = self.cfg
        n = cfg.eval_grid
        xs = torch.linspace(0., 1., n, device=device)
        ys = torch.linspace(0., 1., n, device=device)
        xd = torch.linspace(0., 1., n, device=device)
        yd = torch.linspace(-1., 0., n, device=device)

        Xs, Ys = torch.meshgrid(xs, ys, indexing="ij")
        Xd, Yd = torch.meshgrid(xd, yd, indexing="ij")
        xs = Xs.reshape(-1, 1).requires_grad_(True)
        ys = Ys.reshape(-1, 1).requires_grad_(True)
        xd = Xd.reshape(-1, 1).requires_grad_(True)
        yd = Yd.reshape(-1, 1).requires_grad_(True)

        u_s, v_s, p_s = self.model_s(xs, ys)
        u_d, v_d, p_d = self.model_d(xd, yd)

        def center(p):
            return p - torch.mean(p)

        p_s = center(p_s); p_d = center(p_d)
        p_s_ex = center(self.Exact.p(xs, ys))
        p_d_ex = center(self.Exact.p(xd, yd))
        u_s_ex = self.Exact.u(xs, ys); v_s_ex = self.Exact.v(xs, ys)
        u_d_ex = self.Exact.u(xd, yd); v_d_ex = self.Exact.v(xd, yd)

        return {
            "L2_u_s": (torch.sum((u_s - u_s_ex) ** 2) / (torch.sum(u_s_ex ** 2) + 1e-14)) ** 0.5,
            "L2_v_s": (torch.sum((v_s - v_s_ex) ** 2) / (torch.sum(v_s_ex ** 2) + 1e-14)) ** 0.5,
            "L2_p_s": (torch.sum((p_s - p_s_ex) ** 2) / (torch.sum(p_s_ex ** 2) + 1e-14)) ** 0.5,
            "L2_u_d": (torch.sum((u_d - u_d_ex) ** 2) / (torch.sum(u_d_ex ** 2) + 1e-14)) ** 0.5,
            "L2_v_d": (torch.sum((v_d - v_d_ex) ** 2) / (torch.sum(v_d_ex ** 2) + 1e-14)) ** 0.5,
            "L2_p_d": (torch.sum((p_d - p_d_ex) ** 2) / (torch.sum(p_d_ex ** 2) + 1e-14)) ** 0.5,
        }

    def log_metrics(self, epoch, phase, loss_s, loss_d):
        metrics = self.compute_metrics()
        metrics = {k: v.detach().cpu().item() for k, v in metrics.items()}
        metrics["L2_sum"] = sum(metrics.values())
        self.history["epoch"].append(epoch if phase == "Adam" else self.cfg.epochs_adam + epoch)
        self.history["loss_s"].append(loss_s)
        self.history["loss_d"].append(loss_d)
        for k in ["L2_u_s", "L2_p_s", "L2_u_d", "L2_p_d", "L2_sum"]:
            self.history[k].append(metrics[k])
        print(f"Epoch {epoch:5d} [{phase}] | loss_s={loss_s:.4e} | loss_d={loss_d:.4e} | "
              f"L2_u_s={metrics['L2_u_s']:.2e} | L2_p_s={metrics['L2_p_s']:.2e} | "
              f"L2_u_d={metrics['L2_u_d']:.2e} | L2_p_d={metrics['L2_p_d']:.2e}")

    def train(self):
        cfg = self.cfg
        print("Training XPINNs Case 1...")

        for epoch in range(1, cfg.epochs_adam + 1):
            self.opt_s.zero_grad()
            loss_s = self.loss_stokes()
            loss_s.backward()
            self.opt_s.step()
            self.scheduler_s.step()

            self.opt_d.zero_grad()
            loss_d = self.loss_darcy()
            loss_d.backward()
            self.opt_d.step()
            self.scheduler_d.step()

            if epoch % 500 == 0 or epoch == 1:
                self.log_metrics(epoch, "Adam", loss_s.item(), loss_d.item())

        if cfg.use_lbfgs:
            print("--- Starting L-BFGS ---")
            for epoch in range(1, cfg.epochs_lbfgs + 1):
                def closure_s():
                    self.lbfgs_s.zero_grad()
                    ls = self.loss_stokes()
                    ls.backward()
                    return ls

                def closure_d():
                    self.lbfgs_d.zero_grad()
                    ld = self.loss_darcy()
                    ld.backward()
                    return ld

                self.lbfgs_s.step(closure_s)
                self.lbfgs_d.step(closure_d)
                if epoch % 20 == 0 or epoch == 1:
                    ls = self.loss_stokes().item()
                    ld = self.loss_darcy().item()
                    self.log_metrics(epoch, "LBFGS", ls, ld)

        return self.history

# ============================================================
# VISUALISATION
# ============================================================

def plot_convergence(history, cfg):
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(epochs, history["loss_s"], label='Stokes loss')
    ax1.semilogy(epochs, history["loss_d"], label='Darcy loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Losses'); ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    ax2.plot(epochs, history["L2_u_s"], label='L2 $u_s$')
    ax2.plot(epochs, history["L2_p_s"], label='L2 $p_s$')
    ax2.plot(epochs, history["L2_u_d"], label='L2 $u_d$')
    ax2.plot(epochs, history["L2_p_d"], label='L2 $p_d$')
    ax2.plot(epochs, history["L2_sum"], label='L2 sum', lw=2, color='k')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Relative L2 error (linear scale)')
    ax2.set_title('Error Evolution (Linear Scale)'); ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(max(0, cfg.epochs_adam - 2000), epochs[-1])

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/convergence_case1.png", dpi=cfg.dpi, bbox_inches='tight')
    plt.close()

def plot_global_fields(solver, cfg):
    n = cfg.eval_grid
    x = torch.linspace(0, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xg = X.reshape(-1, 1).requires_grad_(True)
    yg = Y.reshape(-1, 1).requires_grad_(True)

    mask_s = (yg >= 0).squeeze()
    mask_d = (yg < 0).squeeze()

    u_pred = torch.zeros_like(xg); v_pred = torch.zeros_like(xg); p_pred = torch.zeros_like(xg)
    if mask_s.any():
        u_s, v_s, p_s = solver.model_s(xg[mask_s], yg[mask_s])
        u_pred[mask_s] = u_s; v_pred[mask_s] = v_s; p_pred[mask_s] = p_s
    if mask_d.any():
        u_d, v_d, p_d = solver.model_d(xg[mask_d], yg[mask_d])
        u_pred[mask_d] = u_d; v_pred[mask_d] = v_d; p_pred[mask_d] = p_d

    u_ex = solver.Exact.u(xg, yg)
    v_ex = solver.Exact.v(xg, yg)
    p_ex = solver.Exact.p(xg, yg)

    def to_np(t):
        return t.detach().cpu().numpy().reshape(n, n).T

    U_pred, U_ex = to_np(u_pred), to_np(u_ex); U_err = np.abs(U_pred - U_ex)
    V_pred, V_ex = to_np(v_pred), to_np(v_ex); V_err = np.abs(V_pred - V_ex)
    P_pred, P_ex = to_np(p_pred), to_np(p_ex); P_err = np.abs(P_pred - P_ex)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    variables = [('u', U_pred, U_ex, U_err), ('v', V_pred, V_ex, V_err), ('p', P_pred, P_ex, P_err)]

    x_line = np.linspace(0, 1, 200)
    y_line = np.zeros_like(x_line)

    for row, (var, pred, ex, err) in enumerate(variables):
        cmap = 'coolwarm' if var == 'p' else 'viridis'
        im0 = axes[row, 0].imshow(pred, origin='lower', extent=[0, 1, -1, 1], cmap=cmap)
        axes[row, 0].set_title(f"{var} predicted"); plt.colorbar(im0, ax=axes[row, 0])
        im1 = axes[row, 1].imshow(ex, origin='lower', extent=[0, 1, -1, 1], cmap=cmap)
        axes[row, 1].set_title(f"{var} exact"); plt.colorbar(im1, ax=axes[row, 1])
        im2 = axes[row, 2].imshow(err, origin='lower', extent=[0, 1, -1, 1], cmap='hot',
                                   vmin=0, vmax=err.max())
        axes[row, 2].set_title(f"|{var} error| = |exact - pred| (linear)")
        plt.colorbar(im2, ax=axes[row, 2])

    for ax in axes.flat:
        ax.plot(x_line, y_line, color='white', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/global_fields_case1.png", dpi=cfg.dpi, bbox_inches='tight')
    plt.close()

def plot_region_fields(solver, cfg, region='stokes'):
    n = cfg.eval_grid
    if region == 'stokes':
        x = torch.linspace(0, 1, n, device=device)
        y = torch.linspace(0, 1, n, device=device)
        model = solver.model_s
        title_suffix = "Stokes"
    else:
        x = torch.linspace(0, 1, n, device=device)
        y = torch.linspace(-1, 0, n, device=device)
        model = solver.model_d
        title_suffix = "Darcy"

    X, Y = torch.meshgrid(x, y, indexing='ij')
    xg = X.reshape(-1, 1).requires_grad_(True)
    yg = Y.reshape(-1, 1).requires_grad_(True)

    u, v, p = model(xg, yg)
    u_ex = solver.Exact.u(xg, yg)
    v_ex = solver.Exact.v(xg, yg)
    p_ex = solver.Exact.p(xg, yg)

    def to_np(t):
        return t.detach().cpu().numpy().reshape(n, n).T

    U_pred, U_ex = to_np(u), to_np(u_ex); U_err = np.abs(U_pred - U_ex)
    V_pred, V_ex = to_np(v), to_np(v_ex); V_err = np.abs(V_pred - V_ex)
    P_pred, P_ex = to_np(p), to_np(p_ex); P_err = np.abs(P_pred - P_ex)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    variables = [('u', U_pred, U_ex, U_err), ('v', V_pred, V_ex, V_err), ('p', P_pred, P_ex, P_err)]

    x_line = np.linspace(0, 1, 200)
    y_line = np.zeros_like(x_line)
    y_min = 0 if region == 'stokes' else -1
    y_max = 1 if region == 'stokes' else 0

    for row, (var, pred, ex, err) in enumerate(variables):
        cmap = 'coolwarm' if var == 'p' else 'viridis'
        im0 = axes[row, 0].imshow(pred, origin='lower', extent=[0, 1, y_min, y_max], cmap=cmap)
        axes[row, 0].set_title(f"{var} predicted - {title_suffix}"); plt.colorbar(im0, ax=axes[row, 0])
        im1 = axes[row, 1].imshow(ex, origin='lower', extent=[0, 1, y_min, y_max], cmap=cmap)
        axes[row, 1].set_title(f"{var} exact"); plt.colorbar(im1, ax=axes[row, 1])
        im2 = axes[row, 2].imshow(err, origin='lower', extent=[0, 1, y_min, y_max], cmap='hot',
                                   vmin=0, vmax=err.max())
        axes[row, 2].set_title(f"|{var} error| = |exact - pred| (linear)")
        plt.colorbar(im2, ax=axes[row, 2])

    for ax in axes.flat:
        ax.plot(x_line, y_line, color='white', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/fields_{region}_case1.png", dpi=cfg.dpi, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN
# ============================================================

def run():
    cfg = CFG
    print("\n" + "=" * 60)
    print("  CAS 1")
    print("=" * 60)

    solver = XPINNSolver(cfg)
    history = solver.train()

    final_metrics = solver.compute_metrics()
    with open(f"{cfg.output_dir}/final_metrics_case1.txt", "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v.detach().cpu().item():.6e}\n")

    plot_convergence(history, cfg)
    plot_global_fields(solver, cfg)
    plot_region_fields(solver, cfg, 'stokes')
    plot_region_fields(solver, cfg, 'darcy')

    print(f"Done. Figures saved in {cfg.output_dir}")
    return history, final_metrics

if __name__ == "__main__":
    run()
