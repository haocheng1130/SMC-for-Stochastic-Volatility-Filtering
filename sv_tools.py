"""Stochastic volatility with return-volatility correlation: bootstrap particle filter, simulated likelihood,
GARCH benchmarks.

Model, with theta = (a, mu, phi, sigma_x, rho):
    R_j     = a + exp(X_j / 2) eta_j
    X_{j+1} = mu + phi (X_j - mu) + sigma_x xi_{j+1},   X stationary, sd sigma_x / sqrt(1 - phi^2).
Two timings of the correlation:
    "next":  corr(eta_j, xi_{j+1}) = rho   (today's return moves tomorrow's volatility: leverage)
    "same":  corr(eta_j, xi_j) = rho       (today's return and today's volatility move together)
rho = 0 is the model without correlation, identical under both timings.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

LOG_2PI = np.log(2.0 * np.pi)


def same_day_step(x_prev, resid, theta, u, proposal):
    """One step of the same-day model from resampled particles x_prev of X_{j-1}, the residual r_j - a and standard
    normal draws u. Returns the proposed shocks xi, the new states x and the log weights logw, with the pieces
    the filter reuses (mean_x, dev, inv_var)."""
    a, mu, phi, sx, rho = theta
    s_rho = np.sqrt(1.0 - rho ** 2)
    mean_x = mu + phi * (x_prev - mu)
    if proposal == "guided":
        xi = rho * resid * np.exp(-0.5 * mean_x) + s_rho * u
        log_ratio = 0.5 * (u * u - xi * xi) + np.log(s_rho)   # log N(xi; 0, 1) - log q(xi)
    else:
        xi, log_ratio = u, 0.0
    x = mean_x + sx * xi
    dev = resid - rho * np.exp(0.5 * x) * xi
    log_var = 2.0 * np.log(s_rho) + x
    inv_var = np.exp(-log_var)
    logw = -0.5 * (LOG_2PI + log_var + dev * dev * inv_var) + log_ratio
    return {"xi": xi, "x": x, "logw": logw, "mean_x": mean_x, "dev": dev, "inv_var": inv_var}


def particle_filter(r, theta, N, seed, timing="next", proposal="bootstrap", store=False):
    """Particle filter. Returns the log-likelihood estimate and, if store, per-day summaries.

    The random numbers are drawn in the same order for every theta, so a fixed seed gives common random
    numbers across parameter values. Particles are sorted before systematic resampling, which keeps the
    likelihood estimate close to continuous in theta for this one-dimensional state.

    "next": propagation uses eta_j of each resampled particle, xi_{j+1} = rho eta_j + sqrt(1 - rho^2) zeta;
            R_j | X_j ~ N(a, e^{X_j}).
    "same": propagation draws xi_j, and R_j | X_{j-1}, xi_j ~ N(a + rho e^{X_j/2} xi_j, (1 - rho^2) e^{X_j}).
            proposal="bootstrap" draws xi_j ~ N(0, 1). proposal="guided" draws xi_j ~ N(rho z_j, 1 - rho^2),
            z_j = (r_j - a) e^{-m_j/2}, m_j = mu + phi (X_{j-1} - mu): the exact conditional of xi_j given
            r_j when sigma_x = 0. The weights carry the ratio N(xi_j; 0, 1) / N(xi_j; rho z_j, 1 - rho^2).
    """
    a, mu, phi, sx, rho = theta
    rng = np.random.default_rng(seed)
    T = len(r)
    s_rho = np.sqrt(1.0 - rho ** 2)
    x = mu + sx / np.sqrt(1.0 - phi ** 2) * rng.standard_normal(N)   # X_1 ("next") or X_0 ("same")
    loglik = np.empty(T)
    if store:
        out = {k: np.empty(T) for k in ("pred_var", "pred_vol", "pit", "filt_logvar", "filt_var", "filt_vol",
                                        "ess")}
        rng_pred = np.random.default_rng([seed, 1])   # separate stream, so storing does not change the filter
    positions = np.arange(N) / N
    for j in range(T):
        resid = r[j] - a
        if timing == "same":
            step = same_day_step(x, resid, theta, rng.standard_normal(N), proposal)
            x, logw, mean_x, dev, inv_var = step["x"], step["logw"], step["mean_x"], step["dev"], step["inv_var"]
        else:
            dev, inv_var = resid, np.exp(-x)
            logw = -0.5 * (LOG_2PI + x + dev * dev * inv_var)
        m = logw.max()
        if not np.isfinite(m):
            return {"loglik": -np.inf}
        w = np.exp(logw - m)
        sw = w.sum()
        loglik[j] = m + np.log(sw / N)
        w /= sw
        if store:
            # one-step predictive quantities given r_{1:j-1}; pred_var forecasts e^{X_j}
            ex = np.exp(x)
            if timing == "same":   # E[e^{X_j} | X_{j-1}] in closed form, valid for either proposal
                out["pred_var"][j] = np.mean(np.exp(mean_x + 0.5 * sx * sx))
                out["pred_vol"][j] = np.mean(np.exp(0.5 * mean_x + 0.125 * sx * sx))
            else:
                out["pred_var"][j] = np.mean(ex)
                out["pred_vol"][j] = np.mean(np.sqrt(ex))
            if proposal == "guided" and timing == "same":
                # the PIT needs xi_j from N(0, 1): reweighting the guided draws by N(xi; 0, 1) / q(xi) alone has
                # infinite variance when 1 - rho^2 < 1/2
                xp = rng_pred.standard_normal(N)
                x_p = mean_x + sx * xp
                out["pit"][j] = np.mean(ndtr((resid - rho * np.exp(0.5 * x_p) * xp) * np.exp(-0.5 * x_p) / s_rho))
            else:
                out["pit"][j] = np.mean(ndtr(dev * np.sqrt(inv_var)))
            out["filt_logvar"][j] = w @ x
            out["filt_var"][j] = w @ ex
            out["filt_vol"][j] = w @ np.sqrt(ex)
            out["ess"][j] = 1.0 / (w @ w)
        order = np.argsort(x)
        cs = np.cumsum(w[order])
        cs[-1] = 1.0
        x = x[order][np.searchsorted(cs, rng.random() / N + positions)]
        if timing != "same":
            # xi_{j+1} = rho eta_j + sqrt(1 - rho^2) zeta_{j+1}, with eta_j of each resampled particle
            eta = resid * np.exp(-0.5 * x)
            x = mu + phi * (x - mu) + sx * (rho * eta + s_rho * rng.standard_normal(N))
    # particles: resampled X_{T-1} for "same", propagated X_T for "next"
    res = {"loglik": float(loglik.sum()), "loglik_t": loglik, "particles": x}
    if store:
        res.update(out)
    return res


def to_theta(z, leverage=True):
    """Unconstrained z -> theta = (a, mu, phi, sigma_x, rho); a is scaled by 1e-3 for the optimizer."""
    rho = np.tanh(z[4]) if leverage else 0.0
    return np.array([1e-3 * z[0], z[1], np.tanh(z[2]), np.exp(z[3]), rho])


def from_theta(theta, leverage=True):
    a, mu, phi, sx, rho = theta
    z = [1e3 * a, mu, np.arctanh(phi), np.log(sx)]
    return np.array(z + [np.arctanh(rho)] if leverage else z)


def fit_sv(r, N, seed, leverage=True, timing="next", proposal="bootstrap", maxfev=3000):
    """Maximize the particle log-likelihood (common random numbers), started from the best point of a grid."""
    def nll(z):
        theta = to_theta(z, leverage)
        if not (0.0 < theta[2] < 0.9999 and theta[3] < 2.0 and abs(theta[4]) < 0.99):
            return 1e10
        ll = particle_filter(r, theta, N, seed, timing, proposal)["loglik"]
        return -ll if np.isfinite(ll) else 1e10

    var = np.var(r)
    starts = []
    for phi in (0.95, 0.98, 0.99):
        for sx in (0.1, 0.2, 0.3):
            for rho in ((-0.6, -0.3, 0.0) if leverage else (0.0,)):
                s2 = sx ** 2 / (1 - phi ** 2)
                starts.append(from_theta((np.mean(r), np.log(var) - s2 / 2, phi, sx, rho), leverage))
    values = [nll(z) for z in starts]
    z0 = starts[int(np.argmin(values))]
    res = minimize(nll, z0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-9, "maxfev": maxfev})
    return {"theta": to_theta(res.x, leverage), "loglik": -res.fun, "nfev": res.nfev, "success": res.success,
            "grid_best": to_theta(z0, leverage), "grid_loglik": -min(values)}


def garch_filter(r, params, h0):
    """GJR-GARCH(1,1) with constant mean; gamma = 0 gives GARCH(1,1). Returns predictive variances and
    one-step log predictive densities."""
    a, omega, alpha, beta, gamma = params
    eps = r - a
    h = np.empty(len(r))
    h[0] = h0
    for j in range(1, len(r)):
        h[j] = omega + (alpha + gamma * (eps[j - 1] < 0)) * eps[j - 1] ** 2 + beta * h[j - 1]
    return h, -0.5 * (LOG_2PI + np.log(h) + eps ** 2 / h)


def fit_garch(r, gjr):
    """Gaussian maximum likelihood; h_1 is the sample variance of r."""
    h0 = np.var(r)

    def unpack(x):
        return np.array([1e-3 * x[0], 1e-6 * x[1], x[2], x[3], x[4] if gjr else 0.0])

    def nll(x):
        p = unpack(x)
        if p[1] <= 0 or p[2] < 0 or p[3] < 0 or p[2] + p[4] < 0 or p[2] + p[3] + p[4] / 2 >= 1:
            return 1e10
        return -garch_filter(r, p, h0)[1].sum()

    x0 = [1e3 * np.mean(r), 1e6 * h0 * 0.05, 0.05, 0.9] + ([0.05] if gjr else [0.0])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-8, "maxfev": 20000, "maxiter": 20000})
    return {"params": unpack(res.x), "loglik": -res.fun, "success": res.success, "h0": h0}


def forecast_scores(r, rv, pred_var, loglik_t, pit, mask):
    """Scores of one-step forecasts on the days in mask. QLIKE compares the predictive variance with RV."""
    q = rv[mask] / pred_var[mask]
    u = pit[mask]
    return {
        "days": int(mask.sum()),
        "loglik": float(loglik_t[mask].sum()),
        "qlike": float(np.mean(q - np.log(q) - 1.0)),
        "corr_vol_rv": float(np.corrcoef(np.sqrt(pred_var[mask]), np.sqrt(rv[mask]))[0, 1]),
        "pit_below_1pct": float(np.mean(u < 0.01)),
        "pit_above_99pct": float(np.mean(u > 0.99)),
        "pit_below_5pct": float(np.mean(u < 0.05)),
        "pit_above_95pct": float(np.mean(u > 0.95)),
        "beyond_3sd": float(np.mean((u < ndtr(-3.0)) | (u > ndtr(3.0)))),   # nominal 0.0027
    }
