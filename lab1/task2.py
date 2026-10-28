"""КП №1, завдання 2: часова затримка сортування колекції 1 000 000 об'єктів.

Запуск:  python task2_simple.py                    (вимірювання + аналіз)
         python task2_simple.py sort_times.csv     (аналіз збережених вимірювань)
"""
import sys, time, gc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N_ITEMS = 1_000_000     # розмір колекції
REPEATS = 200           # скільки разів вимірюємо
WARMUP = 5              # прогрівальні запуски (не записуються)
DISCARD = 40            # перші записані запуски відкидаємо: час спочатку «виходить на режим»
ALPHA = 0.05


class Item:
    __slots__ = ("v",)
    def __init__(self, v):
        self.v = v


# 1. ВИМІРЮВАННЯ ------------------------------------------------------------
def measure_sorting_times(repeats=REPEATS, warmup=WARMUP):
    """Сортує нову випадкову колекцію з 10^6 об'єктів; вимірюється лише сортування."""
    rng = np.random.default_rng(2026)
    build = lambda: [Item(v) for v in rng.integers(0, 10**9, N_ITEMS).tolist()]
    for _ in range(warmup):
        d = build(); d.sort(key=lambda o: o.v)
    times = []
    for i in range(repeats):
        data = build()
        gc.collect()
        t0 = time.perf_counter()
        data.sort(key=lambda o: o.v)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 20 == 0:
            print(f"  виконано {i + 1}/{repeats}", flush=True)
    times = np.array(times)
    np.savetxt("sort_times.csv", times, header="time_s", comments="")   # збереження
    return times


# 2. СЕРЕДНЄ І ДИСПЕРСІЯ ----------------------------------------------------
def calculate_statistics(x, title):
    mean, var = x.mean(), x.var(ddof=1)
    print(f"{title}: n = {len(x)}, середнє = {mean:.6f} с, дисперсія = {var:.3e}, "
          f"σ = {np.sqrt(var):.6f} с, min = {x.min():.6f}, max = {x.max():.6f}")
    return mean, var


# 3. ГІСТОГРАМА ------------------------------------------------------------
def plot_histogram(x, title="Гістограма часу сортування"):
    plt.figure(figsize=(8, 4.5))
    plt.hist(x, bins=20, edgecolor="black", alpha=.75)
    plt.title(title); plt.xlabel("Час, с"); plt.ylabel("Кількість вимірювань")
    plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()


# 4. ІДЕНТИФІКАЦІЯ ЗАКОНУ (χ²) ---------------------------------------------
def chi_square_test(x, dist, params, n_params):
    """χ² з рівноймовірними інтервалами: межі — квантили закону, очікувана
    частота n/k в кожному; df = k - 1 - кількість оцінених параметрів."""
    n = len(x); k = min(20, n // 5)
    df = k - 1 - n_params
    edges = np.concatenate(([-np.inf], dist.ppf(np.arange(1, k) / k, *params), [np.inf]))
    obs, _ = np.histogram(x, bins=edges)
    stat = np.sum((obs - n / k) ** 2 / (n / k))
    return stat, stats.chi2.ppf(1 - ALPHA, df), df


def identify_distribution(x):
    """Підбирає параметри (метод максимальної правдоподібності) і перевіряє кожен закон за χ².
    Серед тих, що пройшли перевірку, обирає найменший AIC."""
    m, s = x.mean(), x.std(ddof=1)
    candidates = [   # (назва, закон, параметри, кількість параметрів)
        ("Нормальний", stats.norm, stats.norm.fit(x), 2),
        ("Експоненційний зі зсувом", stats.expon, stats.expon.fit(x), 2),
        ("Логнормальний зі зсувом", stats.lognorm, stats.lognorm.fit(x, 0.5, loc=x.min() - s, scale=s), 3),
        ("Гамма зі зсувом", stats.gamma, stats.gamma.fit(x, 4, loc=x.min() - s, scale=s / 2), 3),
        ("Вейбулла зі зсувом", stats.weibull_min, stats.weibull_min.fit(x, 1.5, loc=x.min() - 0.1 * s, scale=1.5 * s), 3),
    ]
    print(f"\n{'Розподіл':<27}{'AIC':>10}{'χ²':>9}{'χ²_кр':>8}{'df':>4}  результат")
    rows = []
    for name, dist, params, k in candidates:
        aic = 2 * k - 2 * np.sum(dist.logpdf(x, *params))
        stat, crit, df = chi_square_test(x, dist, params, k)
        ok = stat < crit
        rows.append((name, dist, params, aic, stat, crit, ok))
        print(f"{name:<27}{aic:>10.1f}{stat:>9.2f}{crit:>8.2f}{df:>4}  {'приймається' if ok else 'відхиляється'}")
    accepted = [r for r in rows if r[6]]
    if accepted:
        best = min(accepted, key=lambda r: r[3])
        print(f"\nОбрано: {best[0]} (серед прийнятих — найменший AIC)")
    else:
        best = min(rows, key=lambda r: r[4] / r[5])
        print(f"\nУВАГА: жоден закон не пройшов χ². Береться найближчий: {best[0]} — імітація буде наближеною.")
    print("Параметри (форма, зсув, масштаб):", np.round(best[2], 6))
    return best[0], best[1], best[2]


# 5. ГЕНЕРАТОР ЗАТРИМОК ----------------------------------------------------
def generate_delays(dist, params, count=10_000, seed=123):
    """Метод оберненої функції: r = F^(-1)(ζ), ζ — рівномірне на (0;1)."""
    zeta = np.clip(np.random.default_rng(seed).random(count), 1e-12, 1 - 1e-12)
    return dist.ppf(zeta, *params)


# 6. ПОРІВНЯННЯ РЕАЛЬНИХ І ЗГЕНЕРОВАНИХ ДАНИХ ------------------------------
def compare_times(real, sim):
    print("\nПОРІВНЯННЯ")
    print(f"{'':<14}{'реальні':>11}{'згенеровані':>13}{'похибка, %':>12}")
    pairs = [("середнє", real.mean(), sim.mean()), ("σ", real.std(ddof=1), sim.std(ddof=1)),
             ("медіана", np.median(real), np.median(sim)),
             ("P95", np.percentile(real, 95), np.percentile(sim, 95))]
    for name, a, b in pairs:
        print(f"{name:<14}{a:>11.6f}{b:>13.6f}{100 * (b - a) / a:>12.2f}")
    ks = stats.ks_2samp(real, sim)
    print(f"KS-тест двох вибірок: D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}")
    mean_err = abs(pairs[0][2] - pairs[0][1]) / pairs[0][1] * 100
    good = mean_err < 1 and ks.pvalue > ALPHA
    print("ВИСНОВОК: " + ("генератор відтворює затримку задовільно (середнє < 1 %, KS p > 0,05)."
                          if good else "точність недостатня — потрібно уточнити модель."))
    plt.figure(figsize=(8, 4.5))
    plt.hist(real, bins=20, density=True, alpha=.6, edgecolor="black", label="виміряні")
    plt.hist(sim, bins=60, density=True, histtype="step", color="r", label="згенеровані")
    plt.title("Реальні та згенеровані затримки"); plt.xlabel("Час, с"); plt.ylabel("Щільність")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()


# ГОЛОВНА ПРОГРАМА ----------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        times = np.loadtxt(sys.argv[1], skiprows=1)
    else:
        print(f"Вимірювання сортування {N_ITEMS:,} об'єктів, {REPEATS} запусків")
        times = measure_sorting_times()

    calculate_statistics(times, "Усі запуски")
    x = times[DISCARD:]                                   # без перших запусків («вихід на режим»)
    q1, q3 = np.percentile(x, [25, 75])
    thr = q3 + 3 * (q3 - q1)                              # межа викидів (Тьюкі)
    n_out = int(np.sum(x > thr))
    x = x[x <= thr]
    print(f"Відкинуто перші {DISCARD} запусків і {n_out} викидів (> {thr:.5f} с, рідкісні затримки ОС).")
    calculate_statistics(x, "Для аналізу")
    plot_histogram(x)

    name, dist, params = identify_distribution(x)
    sim = generate_delays(dist, params)
    calculate_statistics(sim, "Згенеровані")
    compare_times(x, sim)


if __name__ == "__main__":
    main()