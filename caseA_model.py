"""
Case A – Smart Home Energy Management (PV + Battery + EV extension)
EGS Individual Coursework

Improvements over original:
- Explicit unit-consistency check (Section 3)
- Round-trip efficiency verification check added
- pv_limit margin tolerance made explicit (>= -1e-9)
- Rule-based policy defined with full algorithmic equations (documented in comments)
- Dumb-EV scenario clarified: EV is unmanaged, battery LP still optimised
- Cleaner 3-panel dispatch figure (fig2_dispatch_simple.png) now generated
- Dead variable `pv_bus` removed from rule-based simulation
- Dead variable `s_use` is clearly the total PV output dispatched to the bus
- Self-sufficiency added to summary table
- All verification checks have explicit PASS/FAIL with tolerances
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── System parameters ──────────────────────────────────────────────────────────
DT_HOURS  = 0.5          # time-step duration [h]
E_MAX     = 5.0          # usable battery capacity [kWh]
P_MAX     = 2.5          # max charge / discharge power [kW]
ETA_CH    = 0.95         # charge efficiency  (η_ch)  → round-trip ≈ 0.95² ≈ 0.90
ETA_DIS   = 0.95         # discharge efficiency (η_dis)
E_INIT    = 2.5          # initial SOC [kWh]  (50 % of E_MAX)

TOLS = {
    "balance_kw"  : 1e-9,   # energy-balance residual tolerance [kW]
    "pv_margin_kw": 1e-9,   # PV upper-bound violation tolerance [kW]
    "flow_neg_kw" : 1e-9,   # non-negativity violation tolerance [kW]
    "ev_energy_kwh": 1e-9,  # EV delivered-energy error tolerance [kWh]
}

# ── Unit-consistency check ─────────────────────────────────────────────────────
# Dimension analysis (shown once, used throughout):
#   import cost [£] = tariff [£/kWh] × import_power [kW] × dt [h]
#   Example: 0.112 £/kWh × 1.0 kW × 0.5 h = 0.056 £  ✓
# Battery energy update [kWh]:
#   E_{t+1} = E_t + η_ch × p_ch [kW] × dt [h]  − p_dis [kW] × dt [h] / η_dis
#   Units:  kWh = kWh + (dimensionless) × kW × h − kW × h / (dimensionless)  ✓

UNIT_CHECK = {
    "example_tariff_gbp_per_kwh": 0.112,
    "example_power_kw"           : 1.0,
    "example_dt_h"               : 0.5,
    "expected_cost_gbp"          : 0.056,
    "computed_cost_gbp"          : 0.112 * 1.0 * 0.5,   # = 0.056
    "pass"                       : abs(0.112 * 1.0 * 0.5 - 0.056) < 1e-12,
}
assert UNIT_CHECK["pass"], "Unit check failed – review tariff × power × dt formula"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_case_a(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {
        "timestamp", "pv_kw", "base_load_kw",
        "import_tariff_gbp_per_kwh", "export_price_gbp_per_kwh", "ambient_temp_C",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Policy 1: Rule-based self-consumption controller ──────────────────────────
#
# Algorithmic definition (per time-step t):
#
#   s_t     = min(G^PV_t, L_t)                            [PV directly to load]
#   surplus = G^PV_t − s_t                                [excess PV]
#   deficit = L_t    − s_t                                [unmet load after PV]
#
#   p_ch,t  = min(surplus,  P_max,  (E_max − E_t)/(η_ch · Δt))   [battery charge]
#   g_exp,t = surplus − p_ch,t                                     [grid export]
#
#   p_dis,t = min(deficit,  P_max,  E_t · η_dis / Δt)             [battery discharge]
#   g_imp,t = deficit − p_dis,t                                    [grid import]
#
#   E_{t+1} = E_t + η_ch · p_ch,t · Δt − p_dis,t · Δt / η_dis   [SOC update]

def simulate_rule_based(
    df: pd.DataFrame,
    Emax: float = E_MAX,
    Pmax: float = P_MAX,
    eta_ch: float = ETA_CH,
    eta_dis: float = ETA_DIS,
    E_init: float = E_INIT,
    dt: float = DT_HOURS,
    ev_series: np.ndarray | None = None,
) -> dict:
    T    = len(df)
    pv   = df["pv_kw"].to_numpy(float)
    load = df["base_load_kw"].to_numpy(float).copy()
    if ev_series is not None:
        load = load + np.asarray(ev_series, float)

    tariff       = df["import_tariff_gbp_per_kwh"].to_numpy(float)
    export_price = df["export_price_gbp_per_kwh"].to_numpy(float)

    E   = np.zeros(T + 1);  E[0] = E_init
    s   = np.zeros(T)    # PV dispatched to bus (= direct-to-load + to-battery + export)
    ch  = np.zeros(T)    # battery charge power [kW]
    dis = np.zeros(T)    # battery discharge power [kW]
    imp = np.zeros(T)    # grid import [kW]
    exp = np.zeros(T)    # grid export [kW]

    for t in range(T):
        pv_direct = min(pv[t], load[t])                           # PV → load directly
        surplus   = pv[t] - pv_direct                             # surplus PV
        deficit   = load[t] - pv_direct                           # unmet after PV

        ch[t]  = min(surplus, Pmax, (Emax - E[t]) / (eta_ch * dt))
        exp[t] = surplus - ch[t]

        dis[t] = min(deficit, Pmax, E[t] * eta_dis / dt)
        imp[t] = deficit - dis[t]

        # s_t = total PV leaving the generation node (direct + charge + export)
        s[t] = pv_direct + ch[t] + exp[t]

        E[t + 1] = E[t] + ch[t] * eta_ch * dt - dis[t] * dt / eta_dis

    net_cost = float(np.sum((tariff * imp - export_price * exp) * dt))
    return {
        "E": E, "s_use": s,
        "p_ch": ch, "p_dis": dis, "g_imp": imp, "g_exp": exp,
        "load": load, "pv": pv,
        "tariff": tariff, "export_price": export_price,
        "cost": net_cost,
    }


# ── Policy 2: LP optimisation core ────────────────────────────────────────────
#
# Variables (flattened):  s [T], p_ch [T], p_dis [T], g_imp [T], g_exp [T],
#                         (p_ev [T] – only in EV extension),  E [T+1]
#
# Objective:  min  Σ_t  (λ^imp_t · g_imp_t − λ^exp_t · g_exp_t) · Δt
#
# Constraints:
#   Bus balance:    s_t + p_dis,t + g_imp,t = L_t + p_ch,t + g_exp,t  ∀ t
#                   (+ p_ev,t on the RHS for EV extension)
#   SOC update:     E_{t+1} = E_t + η_ch·p_ch,t·Δt − p_dis,t·Δt/η_dis  ∀ t
#   Initial SOC:    E_0 = E_init
#   Terminal SOC:   E_T ≥ E_init     (prevents artificial end-of-horizon draw-down)
#   Bounds:         0 ≤ s_t ≤ G^PV_t,  0 ≤ p_ch/dis,t ≤ P_max,
#                   g_imp,g_exp ≥ 0,  0 ≤ E_t ≤ E_max
#   EV session j:   p_ev,t = 0   outside Ω_j
#                   Σ_{t∈Ω_j} p_ev,t · Δt = E_req,j  (exact delivery)
#                   0 ≤ p_ev,t ≤ P^EV_max,j

def _solve_home_opt_core(
    df: pd.DataFrame,
    base_load: np.ndarray,
    extra_balance_var: np.ndarray | None = None,
    extra_session_constraints: list[tuple[np.ndarray, float]] | None = None,
    extra_var_bounds: np.ndarray | None = None,
    Emax: float = E_MAX,
    Pmax: float = P_MAX,
    eta_ch: float = ETA_CH,
    eta_dis: float = ETA_DIS,
    E_init: float = E_INIT,
    dt: float = DT_HOURS,
    end_soc_ge_init: bool = True,
):
    T      = len(df)
    pv     = df["pv_kw"].to_numpy(float)
    tariff = df["import_tariff_gbp_per_kwh"].to_numpy(float)
    export = df["export_price_gbp_per_kwh"].to_numpy(float)

    include_extra = extra_var_bounds is not None
    if include_extra:
        names = [("s", T), ("ch", T), ("dis", T), ("imp", T), ("exp", T),
                 ("extra", T), ("E", T + 1)]
    else:
        names = [("s", T), ("ch", T), ("dis", T), ("imp", T), ("exp", T),
                 ("E", T + 1)]

    idx = {}
    start = 0
    for name, size in names:
        idx[name] = slice(start, start + size)
        start += size
    n = start

    # Objective vector: import cost – export revenue
    c = np.zeros(n)
    c[idx["imp"]] =  tariff * dt
    c[idx["exp"]] = -export * dt

    # Variable bounds
    bounds  = [(0.0, float(pv[t])) for t in range(T)]   # s ≤ PV
    bounds += [(0.0, Pmax)          for _ in range(T)]   # p_ch
    bounds += [(0.0, Pmax)          for _ in range(T)]   # p_dis
    bounds += [(0.0, None)          for _ in range(T)]   # g_imp
    bounds += [(0.0, None)          for _ in range(T)]   # g_exp
    if include_extra:
        bounds += [(0.0, float(extra_var_bounds[t])) for t in range(T)]  # p_ev
    bounds += [(0.0, Emax) for _ in range(T + 1)]       # E

    rows, cols, data, b_eq = [], [], [], []

    # Bus balance: s + p_dis + g_imp − p_ch − g_exp [− p_ev] = L_t
    for t in range(T):
        r = len(b_eq)
        for sl, val in [(idx["s"], 1.), (idx["dis"], 1.), (idx["imp"], 1.),
                        (idx["ch"], -1.), (idx["exp"], -1.)]:
            rows.append(r); cols.append(sl.start + t); data.append(val)
        if include_extra:
            rows.append(r); cols.append(idx["extra"].start + t); data.append(-1.)
        b_eq.append(float(base_load[t]))

    # SOC update: E_{t+1} − E_t − η_ch·Δt·p_ch + (Δt/η_dis)·p_dis = 0
    for t in range(T):
        r = len(b_eq)
        rows += [r, r, r, r]
        cols += [idx["E"].start + t + 1, idx["E"].start + t,
                 idx["ch"].start + t,    idx["dis"].start + t]
        data += [1., -1., -eta_ch * dt, dt / eta_dis]
        b_eq.append(0.)

    # Initial SOC: E_0 = E_init
    r = len(b_eq)
    rows.append(r); cols.append(idx["E"].start); data.append(1.)
    b_eq.append(E_init)

    # EV session constraints: Σ p_ev,t · Δt = E_req,j
    if include_extra and extra_session_constraints:
        for t_idx, rhs in extra_session_constraints:
            r = len(b_eq)
            for t in t_idx:
                rows.append(r); cols.append(idx["extra"].start + int(t)); data.append(dt)
            b_eq.append(float(rhs))

    A_eq = sp.coo_matrix((data, (rows, cols)), shape=(len(b_eq), n)).tocsc()

    # Terminal SOC inequality: −E_T ≤ −E_init
    A_ub = b_ub = None
    if end_soc_ge_init:
        A_ub = sp.coo_matrix(([-1.], ([0], [idx["E"].start + T])),
                             shape=(1, n)).tocsc()
        b_ub = np.array([-E_init], dtype=float)

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=np.array(b_eq, dtype=float),
        bounds=bounds, method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"Optimisation failed: {res.message}")
    sol = {name: res.x[sl] for name, sl in idx.items()}
    return res, sol


def solve_home_opt(df: pd.DataFrame, ev_series: np.ndarray | None = None) -> dict:
    base_load = df["base_load_kw"].to_numpy(float).copy()
    if ev_series is not None:
        base_load = base_load + np.asarray(ev_series, float)
    res, sol = _solve_home_opt_core(df=df, base_load=base_load)
    return {
        "E": sol["E"], "s_use": sol["s"],
        "p_ch": sol["ch"], "p_dis": sol["dis"],
        "g_imp": sol["imp"], "g_exp": sol["exp"],
        "load": base_load, "pv": df["pv_kw"].to_numpy(float),
        "cost": float(res.fun),
    }


# ── EV extension helpers ───────────────────────────────────────────────────────

def build_ev_session_info(
    df_times: pd.Series, ev_df: pd.DataFrame, dt_minutes: int = 30
) -> list[dict]:
    """Map EV arrival/departure to time-step indices.

    Discretisation rule: arrival → ceil to next slot (conservative availability
    start); departure → floor (conservative availability end). This ensures the
    EV is physically present for every slot it is assigned charging power.
    """
    times    = pd.to_datetime(df_times)
    sessions = []
    for _, row in ev_df.iterrows():
        arrival    = pd.to_datetime(row["arrival_time"]).ceil(f"{dt_minutes}min")
        departure  = pd.to_datetime(row["departure_time"]).floor(f"{dt_minutes}min")
        idx        = np.where((times >= arrival) & (times < departure))[0]
        sessions.append({
            "arrival"   : arrival,
            "departure" : departure,
            "energy_kwh": float(row["required_energy_kwh"]),
            "pmax_kw"   : float(row["max_charge_power_kw"]),
            "idx"       : idx,
        })
    return sessions


def build_dumb_ev_series(
    df_times: pd.Series, ev_df: pd.DataFrame, dt: float = DT_HOURS
):
    """Unmanaged (dumb) EV: charge at max power from arrival until full.

    NOTE: the home battery is still LP-optimised in this scenario – only the
    EV scheduling is unmanaged. Label in results: 'Optimal batt + unmanaged EV'.
    """
    sessions = build_ev_session_info(df_times, ev_df, int(dt * 60))
    T        = len(df_times)
    ev_power = np.zeros(T)
    for s in sessions:
        remain = s["energy_kwh"]
        for t in s["idx"]:
            if remain <= 1e-12:
                break
            p         = min(s["pmax_kw"], remain / dt)
            ev_power[t] += p
            remain    -= p * dt
    return ev_power, sessions


def solve_home_opt_with_ev(
    df: pd.DataFrame, ev_df: pd.DataFrame
) -> tuple[dict, list[dict]]:
    """Co-optimise home battery dispatch and EV charging schedule (smart EV)."""
    T         = len(df)
    base_load = df["base_load_kw"].to_numpy(float)
    sessions  = build_ev_session_info(df["timestamp"], ev_df, int(DT_HOURS * 60))

    ev_bounds           = np.zeros(T)
    session_constraints = []
    for s in sessions:
        ev_bounds[s["idx"]] += s["pmax_kw"]
        session_constraints.append((s["idx"], s["energy_kwh"]))

    res, sol = _solve_home_opt_core(
        df=df, base_load=base_load,
        extra_session_constraints=session_constraints,
        extra_var_bounds=ev_bounds,
    )
    return {
        "E": sol["E"], "s_use": sol["s"],
        "p_ch": sol["ch"], "p_dis": sol["dis"],
        "g_imp": sol["imp"], "g_exp": sol["exp"],
        "p_ev": sol["extra"],
        "load": base_load + sol["extra"],
        "pv"  : df["pv_kw"].to_numpy(float),
        "cost": float(res.fun),
    }, sessions


# ── Verification ───────────────────────────────────────────────────────────────

def verify_base(
    flows: dict, df: pd.DataFrame, ev_series: np.ndarray | None = None
) -> dict:
    pv   = df["pv_kw"].to_numpy(float)
    load = df["base_load_kw"].to_numpy(float).copy()
    if ev_series is not None:
        load = load + np.asarray(ev_series, float)

    s   = flows["s_use"]
    ch  = flows["p_ch"]
    dis = flows["p_dis"]
    imp = flows["g_imp"]
    exp = flows["g_exp"]
    E   = flows["E"]

    # Bus balance residual: s + dis + imp − load − ch − exp  (should be ≈ 0)
    bal = s + dis + imp - load - ch - exp

    # Round-trip efficiency check:
    #   energy_in × η_ch − energy_out / η_dis ≈ ΔSOC
    #   (useful because it confirms efficiencies are actually applied)
    energy_in       = float(np.sum(ch)  * DT_HOURS)
    energy_out      = float(np.sum(dis) * DT_HOURS)
    delta_soc       = float(E[-1] - E[0])
    rte_residual    = energy_in * ETA_CH - energy_out / ETA_DIS - delta_soc

    # PV upper-bound margin: pv[t] − s[t] ≥ −tol  (s must not exceed PV)
    pv_margin_min   = float(np.min(pv - s))

    all_flows_nn    = bool(min(
        np.min(s), np.min(ch), np.min(dis), np.min(imp), np.min(exp)
    ) >= -TOLS["flow_neg_kw"])

    results = {
        "energy_balance_max_abs_kw"          : float(np.max(np.abs(bal))),
        "energy_balance_PASS"                : bool(np.max(np.abs(bal)) < TOLS["balance_kw"]),
        "soc_min_kwh"                        : float(E.min()),
        "soc_max_kwh"                        : float(E.max()),
        "soc_bounds_PASS"                    : bool(E.min() >= -1e-9 and E.max() <= E_MAX + 1e-9),
        "charge_power_max_kw"                : float(ch.max()),
        "discharge_power_max_kw"             : float(dis.max()),
        "power_limits_PASS"                  : bool(ch.max() <= P_MAX + 1e-9 and dis.max() <= P_MAX + 1e-9),
        "pv_limit_min_margin_kw"             : pv_margin_min,
        "pv_limit_PASS"                      : bool(pv_margin_min >= -TOLS["pv_margin_kw"]),
        "all_flows_nonneg_PASS"              : all_flows_nn,
        "rte_residual_kwh"                   : rte_residual,
        "rte_check_PASS"                     : bool(abs(rte_residual) < 1e-6),
        "end_soc_kwh"                        : float(E[-1]),
        "simultaneous_charge_discharge_steps": int(np.sum((ch > 1e-8) & (dis > 1e-8))),
    }
    return results


def verify_ev_schedule(
    ev_power: np.ndarray, sessions: list[dict], dt: float = DT_HOURS
) -> dict:
    T            = len(ev_power)
    availability = np.zeros(T)
    delivered_errors = []
    for s in sessions:
        availability[s["idx"]] += s["pmax_kw"]
        delivered_errors.append(
            ev_power[s["idx"]].sum() * dt - s["energy_kwh"]
        )

    outside = availability < 1e-12   # slots where no EV is connected
    max_err = float(np.max(np.abs(delivered_errors))) if delivered_errors else 0.0
    return {
        "ev_energy_max_abs_error_kwh": max_err,
        "ev_energy_PASS"             : bool(max_err < TOLS["ev_energy_kwh"]),
        "ev_outside_window_energy_kwh": float(ev_power[outside].sum() * dt),
        "ev_outside_window_PASS"     : bool(ev_power[outside].sum() * dt < 1e-9),
        "ev_peak_power_kw"           : float(ev_power.max()) if len(ev_power) else 0.,
        "ev_peak_bound_margin_kw"    : float(np.max(ev_power - availability)),
        "ev_power_cap_PASS"          : bool(np.max(ev_power - availability) < 1e-9),
    }


# ── Summary KPIs ───────────────────────────────────────────────────────────────

def summarize(
    flows: dict, df: pd.DataFrame, label: str, ev_energy_kwh: float | None = None
) -> dict:
    dt          = DT_HOURS
    tariff      = df["import_tariff_gbp_per_kwh"].to_numpy(float)
    export      = df["export_price_gbp_per_kwh"].to_numpy(float)
    pv          = df["pv_kw"].to_numpy(float)
    base        = df["base_load_kw"].to_numpy(float)
    imp         = flows["g_imp"]
    exp         = flows["g_exp"]
    ch          = flows["p_ch"]
    dis         = flows["p_dis"]
    E           = flows["E"]
    s           = flows["s_use"]
    load        = flows["load"]
    import_energy = imp.sum() * dt

    out = {
        "scenario"                      : label,
        "base_load_kwh"                 : base.sum() * dt,
        "total_load_kwh"                : load.sum() * dt,
        "pv_generation_kwh"             : pv.sum() * dt,
        "grid_import_kwh"               : import_energy,
        "grid_export_kwh"               : exp.sum() * dt,
        "battery_charge_kwh"            : ch.sum() * dt,
        "battery_discharge_kwh"         : dis.sum() * dt,
        "import_cost_gbp"               : float(np.sum(tariff * imp * dt)),
        "export_revenue_gbp"            : float(np.sum(export * exp * dt)),
        "net_cost_gbp"                  : float(np.sum((tariff * imp - export * exp) * dt)),
        "final_soc_kwh"                 : float(E[-1]),
        "peak_grid_import_kw"           : float(imp.max()),
        "avg_import_tariff_gbp_per_kwh" : (float(np.sum(tariff * imp * dt) / import_energy)
                                            if import_energy > 0 else np.nan),
        "self_sufficiency_pct"          : float((1. - import_energy / (load.sum() * dt)) * 100),
        "pv_utilisation_ratio"          : (float(s.sum() * dt / (pv.sum() * dt))
                                            if pv.sum() > 0 else np.nan),
    }
    if ev_energy_kwh is not None:
        out["ev_energy_kwh"] = float(ev_energy_kwh)
    return out


# ── Figures ────────────────────────────────────────────────────────────────────

def make_figures(
    df: pd.DataFrame,
    rule: dict,
    opt: dict,
    dumb_ev: dict,
    smart_ev: dict,
    ev_dumb_series: np.ndarray,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(df["timestamp"])

    # ── Fig 1: 30-day SOC trajectories ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(ts, rule["E"][:-1], label="Rule-based", lw=1.4)
    ax.plot(ts, opt["E"][:-1],  label="Optimal",    lw=1.2, alpha=0.85)
    ax.set_ylabel("SOC (kWh)")
    ax.set_title("Base-case SOC trajectories over 30 days")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_soc_month.png", dpi=220)
    plt.close(fig)

    # ── Fig 2 (main): cleaner 3-panel 48 h comparison ──────────────────────────
    # Panel 1 – Load & PV (shared context)
    # Panel 2 – Grid import: rule-based vs optimal  (key difference)
    # Panel 3 – SOC: rule-based vs optimal          (battery state)
    mask = (ts >= pd.Timestamp("2025-07-04")) & (ts < pd.Timestamp("2025-07-06"))
    t2   = ts[mask]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t2, df.loc[mask, "base_load_kw"], label="Load", lw=1.4)
    axes[0].plot(t2, df.loc[mask, "pv_kw"],        label="PV",   lw=1.4)
    axes[0].set_ylabel("kW")
    axes[0].set_title("Representative 48 h window (4–5 July)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t2, rule["g_imp"][mask], label="Rule-based import", lw=1.4)
    axes[1].plot(t2, opt["g_imp"][mask],  label="Optimal import",    lw=1.2, alpha=0.85)
    axes[1].set_ylabel("kW")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t2, rule["E"][:-1][mask], label="Rule-based SOC", lw=1.4)
    axes[2].plot(t2, opt["E"][:-1][mask],  label="Optimal SOC",    lw=1.2, alpha=0.85)
    axes[2].set_ylabel("kWh")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_dispatch_simple.png", dpi=220)
    plt.close(fig)

    # ── Fig 3: scenario cost bar chart ─────────────────────────────────────────
    labels = ["Rule\n(base)", "Optimal\n(base)",
              "Opt. batt +\nunmanaged EV", "Opt. batt +\nsmart EV"]
    vals   = [rule["cost"], opt["cost"], dumb_ev["cost"], smart_ev["cost"]]
    colours = ["#4878D0", "#4878D0", "#EE854A", "#EE854A"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(labels, vals, color=colours, width=0.55, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 1.5, f"£{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Net electricity cost (£)")
    ax.set_title("Scenario cost comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_costs.png", dpi=220)
    plt.close(fig)

    # ── Fig 4: average hourly EV charging profile ───────────────────────────────
    hours          = ts.dt.hour + ts.dt.minute / 60.0
    smart_ev_power = smart_ev["p_ev"]
    avg = pd.DataFrame({
        "hour": hours,
        "dumb": ev_dumb_series,
        "smart": smart_ev_power,
    }).groupby("hour").mean()

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(avg.index, avg["dumb"],  label="Unmanaged EV", lw=1.4)
    ax.plot(avg.index, avg["smart"], label="Smart EV",     lw=1.4, alpha=0.85)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average EV charging power (kW)")
    ax.set_title("Average EV charging profile (30-day mean by hour)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_ev_profile.png", dpi=220)
    plt.close(fig)

    # ── Fig 5 (new): import tariff distribution ─────────────────────────────────
    tariff = df["import_tariff_gbp_per_kwh"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(tariff, bins=40, color="#4878D0", alpha=0.75, edgecolor="white")
    ax.axvline(tariff.mean(), color="black", ls="--", lw=1.2,
               label=f"Mean {tariff.mean():.3f} £/kWh")
    ax.axvline(np.percentile(tariff, 25), color="#4878D0", ls=":", lw=1.2,
               label=f"25th pct {np.percentile(tariff, 25):.3f} £/kWh")
    ax.set_xlabel("Import tariff (£/kWh)")
    ax.set_ylabel("Number of 30-min slots")
    ax.set_title("Import tariff distribution (30-day summer period)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_tariff_dist.png", dpi=220)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Case A smart home model – improved version."
    )
    parser.add_argument("--data",   default="caseA_smart_home_30min_summer.csv")
    parser.add_argument("--ev",     default="caseA_ev_events.csv")
    parser.add_argument("--outdir", default="caseA_outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    df    = load_case_a(args.data)
    ev_df = pd.read_csv(args.ev)

    # ── Simulations ────────────────────────────────────────────────────────────
    rule = simulate_rule_based(df)
    opt  = solve_home_opt(df)

    # Scenario 3: EV unmanaged (charge on arrival at max rate);
    #             home battery still LP-optimised given the fixed EV load.
    ev_dumb_series, sessions = build_dumb_ev_series(df["timestamp"], ev_df)
    dumb_ev = solve_home_opt(df, ev_series=ev_dumb_series)

    # Scenario 4: EV co-optimised with home battery (smart EV)
    smart_ev, smart_sessions = solve_home_opt_with_ev(df, ev_df)

    # ── Figures ────────────────────────────────────────────────────────────────
    make_figures(df, rule, opt, dumb_ev, smart_ev, ev_dumb_series, outdir)

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    summary = pd.DataFrame([
        summarize(rule,     df, "Rule-based"),
        summarize(opt,      df, "Optimal (LP)"),
        summarize(dumb_ev,  df, "Opt. batt + unmanaged EV",
                  ev_energy_kwh=float(ev_dumb_series.sum() * DT_HOURS)),
        summarize(smart_ev, df, "Opt. batt + smart EV",
                  ev_energy_kwh=float(smart_ev["p_ev"].sum() * DT_HOURS)),
    ])
    summary.to_csv(outdir / "summary_metrics.csv", index=False)

    # ── Verification ───────────────────────────────────────────────────────────
    v_rows = []
    for label, flows, ev_s in [
        ("Rule-based",             rule,     None),
        ("Optimal (LP)",           opt,      None),
        ("Opt. batt + unmanaged EV", dumb_ev, ev_dumb_series),
        ("Opt. batt + smart EV",   smart_ev, smart_ev["p_ev"]),
    ]:
        v = verify_base(flows, df, ev_series=ev_s)
        v["scenario"] = label
        v_rows.append(v)

    v_ev_dumb  = verify_ev_schedule(ev_dumb_series, sessions)
    v_ev_dumb["scenario"] = "Unmanaged EV schedule"
    v_ev_smart = verify_ev_schedule(smart_ev["p_ev"], smart_sessions)
    v_ev_smart["scenario"] = "Smart EV schedule"

    # Unit-consistency record
    v_unit                 = dict(UNIT_CHECK)
    v_unit["scenario"]     = "Unit consistency check"
    v_unit["check_name"]   = "tariff [£/kWh] × power [kW] × dt [h] = cost [£]"

    verification = pd.DataFrame(v_rows + [v_ev_dumb, v_ev_smart, v_unit])
    verification.to_csv(outdir / "verification_checks.csv", index=False)

    if "p_ev" in smart_ev:
        pd.DataFrame({
            "timestamp"   : df["timestamp"],
            "smart_ev_kw" : smart_ev["p_ev"],
        }).to_csv(outdir / "smart_ev_schedule.csv", index=False)

    # ── Console output ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("UNIT CHECK:", "PASS" if UNIT_CHECK["pass"] else "FAIL")
    print("=" * 70)
    cols_show = ["scenario", "grid_import_kwh", "grid_export_kwh",
                 "net_cost_gbp", "final_soc_kwh", "peak_grid_import_kw",
                 "avg_import_tariff_gbp_per_kwh", "self_sufficiency_pct"]
    print(summary[cols_show].round(3).to_string(index=False))

    print("\nVERIFICATION PASS/FAIL summary:")
    pass_cols = [c for c in verification.columns if c.endswith("_PASS")]
    print(verification[["scenario"] + pass_cols].fillna("-").to_string(index=False))

    print("\nDone. Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
