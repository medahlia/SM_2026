"""КП №1, завдання 2: часова затримка сортування колекції 10^6 об'єктів.
Запуск:  python task2_fixed.py            (вимірювання 150 разів)
         python task2_fixed.py 300        (вимірювання 300 разів)
         python task2_fixed.py --load sort_times.csv   (готові дані)"""
import sys, time, gc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05


# ============================================================
# 1. ВИМІРЮВАННЯ: 10^6 ОБ'ЄКТІВ, нові випадкові дані в кожному запуску
# ============================================================
class Item:
    __slots__ = ("v",)
    def __init__(self, v):
        self.v = v


def measure_sorting_times(n=1_000_000, repetitions=150, seed=2026):
    rng = np.random.default_rng(seed)

    def build():
        return [Item(v) for v in rng.integers(0, 10**9, n).tolist()]

    warm = build(); warm.sort(key=lambda o: o.v); del warm    # прогрів
    times = []
    for i in range(repetitions):
        data = build()
        gc.collect()                       # збирання сміття не потрапляє у вимірювання
        t0 = time.perf_counter()
        data.sort(key=lambda o: o.v)
        times.append(time.perf_counter() - t0)
        del data
        if (i + 1) % 10 == 0:
            print(f"  виконано {i + 1}/{repetitions}", flush=True)
    return np.array(times)


# ============================================================
# 2. СТАТИСТИКА І ГІСТОГРАМА
# ============================================================
def print_statistics(x, title):
    print(f"\n{title}\n" + "-" * 45)
    print(f"n = {len(x)}, min = {x.min():.6f}, max = {x.max():.6f}")
    print(f"середнє = {x.mean():.6f}, медіана = {np.median(x):.6f}")
    print(f"σ = {x.std(ddof=1):.6f}, дисперсія = {x.var(ddof=1):.3e}")
    print(f"асиметрія = {stats.skew(x):.3f}, ексцес = {stats.kurtosis(x):.3f}")


def plot_histogram(x, title, bins=20):
    plt.figure(figsize=(9, 5))
    plt.hist(x, bins=bins, edgecolor="black", alpha=.75)
    plt.title(title); plt.xlabel("Час сортування, с"); plt.ylabel("Кількість вимірювань")
    plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()


# ============================================================
# 3. КРИТЕРІЇ ЗГОДИ: χ² (n >= 100) і λ-Колмогорова-Смірнова (n < 100)
# ============================================================
def chi2_test(x, cdf, n_params, k=20):
    """Гістограма з k рівних інтервалів, злиття інтервалів з очікуваною
    або спостереженою кількістю < 5, df = k' - 1 - n_params."""
    n = len(x)
    counts, edges = np.histogram(x, bins=k)
    e = edges.copy(); e[0], e[-1] = -np.inf, np.inf          # Σp_i = 1
    exp = n * np.diff(cdf(e))
    obs_m, exp_m, co, ce = [], [], 0, 0.0
    for o, v in zip(counts, exp):
        co += o; ce += v
        if co >= 5 and ce >= 5:
            obs_m.append(co); exp_m.append(ce); co, ce = 0, 0.0
    if co or ce:
        obs_m[-1] += co; exp_m[-1] += ce
    obs_m, exp_m = np.array(obs_m), np.array(exp_m)
    df = len(obs_m) - 1 - n_params
    if df < 1:
        return np.nan, np.nan, df, np.nan
    stat = float(np.sum((obs_m - exp_m) ** 2 / exp_m))
    return stat, float(stats.chi2.ppf(1 - ALPHA, df)), df, float(stats.chi2.sf(stat, df))


def lambda_test(x, cdf):
    """λ = sqrt(n)·max|w_i - w_i^T|, λ_кр = 1,36 (α = 0,05) — для n < 100."""
    xs = np.sort(x); n = len(xs)
    F = cdf(xs)
    d = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(0, n) / n))
    lam = np.sqrt(n) * d
    return float(lam), 1.36


# ============================================================
# 4. ІДЕНТИФІКАЦІЯ ЗАКОНУ (параметри — МНП, у т.ч. зі зсувом)
# ============================================================
def fit_candidates(x):
    m, s, lo = x.mean(), x.std(ddof=1), x.min()
    g = max(stats.skew(x), 0.3)
    a0 = 4 / g**2; sc0 = s / np.sqrt(a0); loc0 = m - a0 * sc0       # початкові значення за моментами
    out = {}
    def add(name, dist, params, k):
        out[name] = (dist, params, k)
    add("Нормальний", stats.norm, stats.norm.fit(x), 2)
    add("Рівномірний", stats.uniform, stats.uniform.fit(x), 2)
    add("Експоненційний зі зсувом", stats.expon, stats.expon.fit(x), 2)
    add("Логнормальний (2)", stats.lognorm, stats.lognorm.fit(x, floc=0), 2)
    add("Гамма (2)", stats.gamma, stats.gamma.fit(x, floc=0), 2)
    add("Вейбулла (2)", stats.weibull_min, stats.weibull_min.fit(x, floc=0), 2)
    try: add("Логнормальний зі зсувом (3)", stats.lognorm, stats.lognorm.fit(x, 0.5, loc=lo - s, scale=s), 3)
    except Exception: pass
    try: add("Гамма зі зсувом (3)", stats.gamma, stats.gamma.fit(x, a0, loc=loc0, scale=sc0), 3)
    except Exception: pass
    try: add("Вейбулла зі зсувом (3)", stats.weibull_min, stats.weibull_min.fit(x, 1.5, loc=lo - 0.1 * s, scale=1.5 * s), 3)
    except Exception: pass
    return out


def identify_distribution(x):
    n = len(x)
    crit_name = "χ²" if n >= 100 else "λ-критерій (n<100)"
    print(f"\nПеревірка згоди: {crit_name}, α = {ALPHA}")
    print(f"{'Розподіл':<30}{'AIC':>11}{'статистика':>12}{'критичне':>10}{'df':>4}{'p':>8}  результат")
    rows = []
    for name, (dist, params, k) in fit_candidates(x).items():
        cdf = lambda t, d=dist, p=params: d.cdf(t, *p)
        aic = 2 * k - 2 * np.sum(dist.logpdf(x, *params))
        if n >= 100:
            st, cr, df, p = chi2_test(x, cdf, k)
        else:
            st, cr = lambda_test(x, cdf); df, p = 0, np.nan
        ok = np.isfinite(st) and st < cr
        rows.append(dict(name=name, dist=dist, params=params, k=k, aic=aic, stat=st, crit=cr, ok=ok))
        print(f"{name:<30}{aic:>11.2f}{st:>12.3f}{cr:>10.3f}{df:>4}{p:>8.4f}  {'приймається' if ok else 'відхиляється'}")
    accepted = [r for r in rows if r["ok"]]
    if not accepted:
        raise RuntimeError("Жоден закон не пройшов критерій згоди — збільшіть кількість вимірювань "
                           "або розширте набір кандидатів.")
    best = min(accepted, key=lambda r: r["aic"])      # серед прийнятих — найменший AIC
    print(f"\nОбрано: {best['name']}  (серед тих, що пройшли критерій згоди, найменший AIC)")
    print("Параметри:", np.round(best["params"], 6))
    return best


# ============================================================
# 5. ГЕНЕРАТОР ЗАТРИМОК: метод оберненої функції  r = F^{-1}(ζ)
# ============================================================
def generate_delays(model, count=10_000, seed=123):
    rng = np.random.default_rng(seed)
    zeta = np.clip(rng.random(count), 1e-12, 1 - 1e-12)         # ζ ~ U(0;1)
    return model["dist"].ppf(zeta, *model["params"])            # обернена функція розподілу


# ============================================================
# 6. ТОЧНІСТЬ ВІДТВОРЕННЯ
# ============================================================
def evaluate_accuracy(real, sim, model, repeats=500, seed=5):
    qs = [5, 25, 50, 75, 95, 99]
    names = ["середнє", "σ", "асиметрія", "min", "max"] + [f"P{q}" for q in qs]
    f = lambda a: [a.mean(), a.std(ddof=1), stats.skew(a), a.min(), a.max()] + list(np.percentile(a, qs))
    r, s = f(real), f(sim)
    print(f"\n{'показник':<10}{'реальні':>11}{'імітація':>11}{'відн. похибка, %':>18}")
    for nm, a, b in zip(names, r, s):
        print(f"{nm:<10}{a:>11.5f}{b:>11.5f}{100 * (b - a) / a:>18.2f}")

    cdf = lambda t: model["dist"].cdf(t, *model["params"])
    st, cr, df, p = chi2_test(sim, cdf, 0)
    print(f"\nχ² імітація vs обраний закон: {st:.3f} (критичне {cr:.3f}, df={df})")
    ks = stats.ks_2samp(real, sim)
    print(f"KS (реальні vs імітація): D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}")
    rng = np.random.default_rng(seed)
    ps = [stats.ks_2samp(real, generate_delays(model, len(real), int(rng.integers(1 << 30)))).pvalue
          for _ in range(repeats)]
    print(f"Вибірки обсягом n={len(real)}, {repeats} повторів: p>0,05 у {100 * np.mean(np.array(ps) > ALPHA):.1f} % випадків")

    mean_err = abs(s[0] - r[0]) / r[0] * 100
    std_err = abs(s[1] - r[1]) / r[1] * 100
    good = mean_err < 1 and std_err < 5 and ks.pvalue > ALPHA
    print("\nВИСНОВОК")
    print("-" * 45)
    print(f"Закон: {model['name']}; похибка середнього {mean_err:.2f} %, σ {std_err:.2f} %, "
          f"KS p = {ks.pvalue:.3f}.")
    print("Імітація відтворює затримку задовільно." if good else
          "Точність недостатня: потрібно уточнити модель (див. таблицю похибок).")

    plt.figure(figsize=(9, 5))
    plt.hist(real, bins=20, density=True, alpha=.6, edgecolor="black", label="виміряні")
    plt.hist(sim, bins=60, density=True, histtype="step", color="r", label="згенеровані")
    plt.title("Реальні та згенеровані затримки"); plt.xlabel("Час, с"); plt.ylabel("Щільність")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()


# ============================================================
# 7. ГОЛОВНА ПРОГРАМА
# ============================================================
def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--load":
        real = np.loadtxt(sys.argv[2], skiprows=1)
    else:
        reps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
        print(f"Вимірювання часу сортування 1 000 000 об'єктів, {reps} запусків")
        real = measure_sorting_times(repetitions=reps)
        np.savetxt("sort_times.csv", real, header="time_s", comments="")
    if len(real) < 100:
        print("УВАГА: n < 100 — χ² незастосовний, використовується λ-критерій; для надійності зробіть ≥ 100 вимірювань.")
    print_statistics(real, "РЕЗУЛЬТАТИ ВИМІРЮВАНЬ")
    plot_histogram(real, "Гістограма часу сортування 10^6 об'єктів")
    best = identify_distribution(real)
    sim = generate_delays(best, 10_000)
    print_statistics(sim, "ЗГЕНЕРОВАНІ ЗАТРИМКИ")
    evaluate_accuracy(real, sim, best)


if __name__ == "__main__":
    main()