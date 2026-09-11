"""Problem 1 (chinchilla_isoflops): 从 data/isoflops_curves.json 拟合 IsoFLOPs 缩放律.

用法:
    python scripts/isoflops_fit.py

输出:
    figures/isoflops_model_size.png   - N_opt vs C 的缩放律 (deliverable a)
    figures/isoflops_dataset_size.png - D_opt vs C 的缩放律 (deliverable b)
    figures/isoflop_profiles.png      - 各预算下的 loss 曲线 + 前沿 loss 缩放律 L_opt(C)
    figures/chinchilla_contour.png    - Chinchilla 参数化拟合的 (N, D) 等高线图
    同时打印 1e23 / 1e24 FLOPs 下的预测值和拟合指数.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "isoflops_curves.json"
FIGURES_DIR = REPO_ROOT / "figures"

TARGET_BUDGETS = [1e23, 1e24]


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def optimal_points(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按计算预算分组, 取每组 final_loss 最低的 run.

    返回 (C, N_opt, D_opt), 其中 D_opt = C / (6 * N_opt).
    """
    best: dict[float, dict] = {}
    for run in runs:
        c = run["compute_budget"]
        if c not in best or run["final_loss"] < best[c]["final_loss"]:
            best[c] = run
    c_arr = np.array(sorted(best))
    n_arr = np.array([best[c]["parameters"] for c in c_arr], dtype=float)
    d_arr = c_arr / (6.0 * n_arr)
    return c_arr, n_arr, d_arr


def fit_power_law(c: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """拟合 y = A * C^exponent, 返回 (exponent, A). 在 log-log 空间做线性回归."""
    log_c = np.log10(c)
    log_y = np.log10(y)
    slope, intercept = np.polyfit(log_c, log_y, 1)
    return slope, 10.0**intercept


def predict(exponent: float, coeff: float, c: float) -> float:
    return coeff * c**exponent


def min_loss_per_budget(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """每个计算预算下最低的 final_loss (前沿 loss)."""
    best: dict[float, float] = {}
    for run in runs:
        c = run["compute_budget"]
        best[c] = min(best.get(c, np.inf), run["final_loss"])
    c_arr = np.array(sorted(best))
    return c_arr, np.array([best[c] for c in c_arr])


def frontier_loss(c: np.ndarray, e: float, k: float, gamma: float) -> np.ndarray:
    """前沿 loss 缩放律: L(C) = E + K * C^(-gamma)."""
    return e + k * c ** (-gamma)


def chinchilla_loss(n: np.ndarray, d: np.ndarray, e: float, a: float, alpha: float, b: float, beta: float) -> np.ndarray:
    """Chinchilla 参数化: L(N, D) = E + A*N^(-alpha) + B*D^(-beta)."""
    return e + a * n ** (-alpha) + b * d ** (-beta)


def fit_chinchilla(runs: list[dict]) -> np.ndarray:
    """用全部 runs 拟合 L = E + A*N^(-alpha) + B*D^(-beta), 返回 (E, A, alpha, B, beta)."""
    n = np.array([r["parameters"] for r in runs], dtype=float)
    c = np.array([r["compute_budget"] for r in runs], dtype=float)
    d = c / (6.0 * n)
    loss = np.array([r["final_loss"] for r in runs])

    def f(x, e, a, alpha, b, beta):
        n_, d_ = x
        return chinchilla_loss(n_, d_, e, a, alpha, b, beta)

    popt, _ = curve_fit(
        f,
        (n, d),
        loss,
        p0=[2.5, 1000.0, 0.3, 1000.0, 0.3],
        bounds=([0.0, 0.0, 0.05, 0.0, 0.05], [np.inf, np.inf, 1.0, np.inf, 1.0]),
        x_scale=[1.0, 1000.0, 0.1, 1000.0, 0.1],
        maxfev=50000,
    )
    return popt


def parametric_frontier(popt: np.ndarray, c_values: np.ndarray, log_n_bounds=(6.0, 13.5)) -> np.ndarray:
    """对每个预算 C 数值最小化 L(N, C/(6N)), 返回数组 [C, N*, L*]."""
    out = []
    for c in c_values:

        def g(log_n: float) -> float:
            n = 10.0**log_n
            d = c / (6.0 * n)
            return float(chinchilla_loss(n, d, *popt))

        res = minimize_scalar(g, bounds=log_n_bounds, method="bounded")
        out.append((c, 10.0**res.x, res.fun))
    return np.array(out)


def plot_chinchilla_contour(
    runs: list[dict],
    popt: np.ndarray,
    frontier_power_law: tuple[float, float, float, float],
    target_predictions: dict[float, tuple[float, float]],
    out_path: Path,
) -> None:
    """(log N, log D) 平面上画参数化拟合的 loss 等高线, 叠加等 FLOP 线和前沿轨迹.

    frontier_power_law = (n_exp, n_coeff, d_exp, d_coeff).
    target_predictions: {C: (N_opt, D_opt)} 目标预算处的幂律预测.
    """
    e_fit, a_fit, alpha_fit, b_fit, beta_fit = popt
    log_n = np.linspace(7.4, 11.6, 240)
    log_d = np.linspace(8.8, 12.2, 240)
    nn, dd = np.meshgrid(10.0**log_n, 10.0**log_d)
    loss_grid = chinchilla_loss(nn, dd, *popt)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    levels = np.linspace(3.0, 8.0, 26)
    cf = ax.contourf(log_n, log_d, loss_grid, levels=levels, cmap="viridis", extend="both")
    fig.colorbar(cf, ax=ax, label="Predicted loss $L(N, D)$")

    cs = ax.contour(log_n, log_d, loss_grid, levels=levels[::5], colors="white", alpha=0.45, linewidths=0.6)
    ax.clabel(cs, fontsize=6.5, fmt="%.1f")

    # 实际训练 run 的位置 (显示证据覆盖范围)
    n_all = np.array([r["parameters"] for r in runs], dtype=float)
    d_all = np.array([r["compute_budget"] for r in runs], dtype=float) / (6.0 * n_all)
    ax.scatter(np.log10(n_all), np.log10(d_all), s=7, color="black", alpha=0.35, label="training runs", zorder=3)

    # 等 FLOP 双曲线 (log D = log(C/6) - log N), 9 个实验预算 + 2 个目标预算
    def draw_isoflop(c_val: float, color: str, linewidth: float, linestyle: str, alpha: float) -> None:
        y_line = np.log10(c_val / 6.0) - log_n
        mask = (y_line >= log_d.min()) & (y_line <= log_d.max())
        if mask.any():
            ax.plot(log_n[mask], y_line[mask], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=2)
        x_top = np.log10(c_val / 6.0) - log_d.max()
        x_bot = np.log10(c_val / 6.0) - log_d.min()
        for x_edge, y_edge in ((x_top, log_d.max()), (x_bot, log_d.min())):
            if log_n.min() + 0.15 <= x_edge <= log_n.max() - 0.6:
                ax.text(
                    x_edge + 0.06,
                    y_edge - 0.05 if y_edge == log_d.max() else y_edge + 0.12,
                    f"{c_val:g}",
                    fontsize=7,
                    color=color,
                    rotation=-38,
                    rotation_mode="anchor",
                )
                break

    budgets = sorted({r["compute_budget"] for r in runs})
    for c_i in budgets:
        draw_isoflop(c_i, color="white", linewidth=0.8, linestyle=":", alpha=0.55)
    for c_t in target_predictions:
        draw_isoflop(c_t, color="crimson", linewidth=1.4, linestyle="--", alpha=0.9)

    # 前沿轨迹: 幂律外推线 + 参数化拟合的数值前沿 + 最低 loss run 的点
    c_grid = np.logspace(np.log10(min(budgets)), 24, 150)
    n_exp, n_coeff, d_exp, _ = frontier_power_law
    n_line = predict(n_exp, n_coeff, c_grid)
    d_line = c_grid / (6.0 * n_line)
    ax.plot(np.log10(n_line), np.log10(d_line), color="tab:red", linestyle="--", linewidth=1.8, label=r"power-law frontier $N_{\mathrm{opt}}(C)$", zorder=4)

    frt = parametric_frontier(popt, c_grid)
    d_frt = c_grid / (6.0 * frt[:, 1])
    ax.plot(np.log10(frt[:, 1]), np.log10(d_frt), color="tab:purple", linestyle=":", linewidth=1.8, label="parametric-fit frontier", zorder=4)

    _, n_opt, d_opt = optimal_points(runs)
    ax.scatter(np.log10(n_opt), np.log10(d_opt), s=60, color="tab:blue", edgecolor="white", linewidth=0.8, marker="o", label="lowest-loss runs", zorder=5)

    for c_t, (n_t, d_t) in target_predictions.items():
        ax.scatter([np.log10(n_t)], [np.log10(d_t)], s=170, color="gold", marker="*", edgecolor="black", linewidth=0.6, zorder=6)
        ax.annotate(
            f"{c_t:.0e}\nN={n_t:.2g}\nD={d_t:.2g}",
            xy=(np.log10(n_t), np.log10(d_t)),
            xytext=(-16, 14),
            textcoords="offset points",
            fontsize=7.5,
            ha="right",
        )

    ax.set_xlim(log_n.min(), log_n.max())
    ax.set_ylim(log_d.min(), log_d.max())
    ax.set_xlabel(r"$\log_{10} N$ (parameters)")
    ax.set_ylabel(r"$\log_{10} D$ (tokens)")
    ax.set_title(
        "Chinchilla parametric fit: "
        rf"$L = {e_fit:.2f} + {a_fit:.0f} N^{{-{alpha_fit:.2f}}} + {b_fit:.0f} D^{{-{beta_fit:.2f}}}$"
    )
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_isoflop_profiles(runs: list[dict], out_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """左图: 各预算下 loss vs 模型规模 (Hoffmann 式二次拟合); 右图: 前沿 loss 缩放律.

    返回 (c_arr, l_opt, popt), popt = (E, K, gamma).
    """
    budgets = sorted({r["compute_budget"] for r in runs})
    norm = mpl.colors.LogNorm(vmin=min(budgets), vmax=max(budgets))
    cmap = plt.get_cmap("viridis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- 左图: IsoFLOP profiles ----
    vertices: list[tuple[float, float, float]] = []
    for c_i in budgets:
        color = cmap(norm(c_i))
        grp = [r for r in runs if r["compute_budget"] == c_i]
        log_n = np.log10([r["parameters"] for r in grp])
        loss = np.array([r["final_loss"] for r in grp])
        order = np.argsort(log_n)
        ax1.scatter(10.0**log_n[order], loss[order], s=26, color=color, zorder=2)

        # Hoffmann 式: 对每个 profile 拟合二次函数 loss = a*logN^2 + b*logN + c
        coef = np.polyfit(log_n, loss, 2)
        if coef[0] > 0:
            grid = np.linspace(log_n.min() - 0.08, log_n.max() + 0.08, 120)
            ax1.plot(10.0**grid, np.polyval(coef, grid), color=color, linestyle="--", alpha=0.65, zorder=1)
            log_n_star = -coef[1] / (2.0 * coef[0])
            l_star = float(np.polyval(coef, log_n_star))
            vertices.append((c_i, float(10.0**log_n_star), l_star))
            ax1.scatter(
                [10.0**log_n_star],
                [l_star],
                s=95,
                color=color,
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Model size $N$ (parameters)")
    ax1.set_ylabel("Final training loss")
    ax1.set_title("IsoFLOP profiles (quadratic fit, stars = minima)")
    ax1.grid(True, which="both", alpha=0.25)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax1,
        label=r"Compute budget $C$ (FLOPs)",
    )

    # ---- 右图: 前沿 loss 缩放律 L_opt(C) ----
    c_arr, l_opt = min_loss_per_budget(runs)
    popt, _ = curve_fit(
        frontier_loss,
        c_arr,
        l_opt,
        p0=[3.0, 20.0, 0.05],
        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0]),
        maxfev=20000,
    )
    e_fit, k_fit, gamma_fit = popt

    if vertices:
        v_c, _, v_l = zip(*vertices)
        ax2.scatter(v_c, v_l, s=40, facecolor="none", edgecolor="tab:purple", marker="o", label="quadratic-fit minima")

    ax2.scatter(c_arr, l_opt, s=55, color="tab:blue", marker="o", label="lowest-loss run (recommended)")
    c_grid = np.logspace(np.log10(c_arr.min()), 24, 200)
    ax2.plot(
        c_grid,
        frontier_loss(c_grid, *popt),
        color="tab:red",
        linestyle="--",
        label=rf"fit: $L = E + K C^{{-\gamma}}$, $\gamma = {gamma_fit:.3f}$",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Compute budget $C$ (FLOPs)")
    ax2.set_ylabel(r"Optimal loss $L_{\mathrm{opt}}(C)$")
    ax2.set_title("Frontier loss scaling law")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved: {out_path}")
    return c_arr, l_opt, popt


def plot_scaling_law(
    c: np.ndarray,
    y: np.ndarray,
    exponent: float,
    coeff: float,
    ylabel: str,
    title: str,
    out_path: Path,
    symbol: str = "N",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    # 原始数据点: 每个预算下的所有 run (浅色) 和最优点 (深色)
    all_runs = load_runs(DATA_PATH)
    c_all = np.array([r["compute_budget"] for r in all_runs])
    n_all = np.array([r["parameters"] for r in all_runs], dtype=float)
    y_all = n_all if ylabel.startswith("Model") else c_all / (6.0 * n_all)
    ax.scatter(c_all, y_all, s=18, color="lightgray", label="all runs", zorder=1)
    ax.scatter(
        c,
        y,
        s=55,
        color="tab:blue",
        marker="o",
        label=rf"${symbol}_{{\mathrm{{opt}}}}(C_i)$",
        zorder=3,
    )

    # 拟合直线 + 外推到 1e24
    c_line = np.logspace(np.log10(c.min()), 24, 200)
    ax.plot(
        c_line,
        predict(exponent, coeff, c_line),
        color="tab:red",
        linestyle="--",
        label=rf"fit: $y \propto C^{{{exponent:.3f}}}$",
        zorder=2,
    )

    # 目标预算处的预测点
    for c_target in TARGET_BUDGETS:
        y_target = predict(exponent, coeff, c_target)
        ax.axvline(c_target, color="gray", alpha=0.4, linewidth=1)
        ax.scatter(
            [c_target],
            [y_target],
            s=80,
            color="tab:orange",
            marker="*",
            zorder=4,
        )
        ax.annotate(
            f"{y_target:.3g}\n@ {c_target:.0e} FLOPs",
            xy=(c_target, y_target),
            xytext=(10, -25),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget $C$ (FLOPs)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    runs = load_runs(DATA_PATH)
    c, n_opt, d_opt = optimal_points(runs)

    print(f"{len(runs)} runs, {len(c)} compute budgets")
    print("\nIsoFLOPs profile (lowest-loss run per budget):")
    print(f"{'C (FLOPs)':>12} {'N_opt':>14} {'D_opt':>14} {'loss':>10}")
    for ci, ni, di in zip(c, n_opt, d_opt):
        loss = min(r["final_loss"] for r in runs if r["compute_budget"] == ci)
        print(f"{ci:>12.1e} {ni:>14.4e} {di:>14.4e} {loss:>10.4f}")

    n_exp, n_coeff = fit_power_law(c, n_opt)
    d_exp, d_coeff = fit_power_law(c, d_opt)
    print(f"\nN_opt = {n_coeff:.4g} * C^{n_exp:.4f}")
    print(f"D_opt = {d_coeff:.4g} * C^{d_exp:.4f}")

    print("\nPredictions:")
    for c_target in TARGET_BUDGETS:
        n_pred = predict(n_exp, n_coeff, c_target)
        d_pred = predict(d_exp, d_coeff, c_target)
        print(f"C = {c_target:.0e}: N_opt = {n_pred:.6g}, D_opt = {d_pred:.6g}")

    FIGURES_DIR.mkdir(exist_ok=True)
    plot_scaling_law(
        c,
        n_opt,
        n_exp,
        n_coeff,
        ylabel=r"Optimal model size $N_{\mathrm{opt}}$",
        title="Compute-optimal model size (IsoFLOPs profiles)",
        out_path=FIGURES_DIR / "isoflops_model_size.png",
        symbol="N",
    )
    plot_scaling_law(
        c,
        d_opt,
        d_exp,
        d_coeff,
        ylabel=r"Optimal dataset size $D_{\mathrm{opt}}$",
        title="Compute-optimal dataset size (IsoFLOPs profiles)",
        out_path=FIGURES_DIR / "isoflops_dataset_size.png",
        symbol="D",
    )
    _, _, (e_fit, k_fit, gamma_fit) = plot_isoflop_profiles(
        runs,
        out_path=FIGURES_DIR / "isoflop_profiles.png",
    )
    print(f"\nFrontier loss fit: L(C) = {e_fit:.4g} + {k_fit:.4g} * C^-{gamma_fit:.4f}")
    for c_target in TARGET_BUDGETS:
        print(f"C = {c_target:.0e}: predicted L_opt = {frontier_loss(c_target, e_fit, k_fit, gamma_fit):.4f}")

    popt = fit_chinchilla(runs)
    e_ch, a_ch, alpha_ch, b_ch, beta_ch = popt
    print(f"\nChinchilla parametric fit (all {len(runs)} runs):")
    print(f"L = {e_ch:.4g} + {a_ch:.4g} * N^-{alpha_ch:.4f} + {b_ch:.4g} * D^-{beta_ch:.4f}")
    print("(Chinchilla paper: alpha ~ 0.34, beta ~ 0.28)")
    impl_n = beta_ch / (alpha_ch + beta_ch)
    print(
        f"implied allocation: N_opt ~ C^{impl_n:.3f}, D_opt ~ C^{1 - impl_n:.3f} "
        f"(IsoFLOPs power law: {n_exp:.3f} / {d_exp:.3f})"
    )
    target_predictions = {
        c_t: (predict(n_exp, n_coeff, c_t), predict(d_exp, d_coeff, c_t)) for c_t in TARGET_BUDGETS
    }
    for c_t, (n_t, d_t) in target_predictions.items():
        l_t = chinchilla_loss(n_t, d_t, *popt)
        print(f"C = {c_t:.0e}: parametric model predicts L(N_opt, D_opt) = {l_t:.4f}")
    plot_chinchilla_contour(
        runs,
        popt,
        frontier_power_law=(n_exp, n_coeff, d_exp, d_coeff),
        target_predictions=target_predictions,
        out_path=FIGURES_DIR / "chinchilla_contour.png",
    )


if __name__ == "__main__":
    main()
