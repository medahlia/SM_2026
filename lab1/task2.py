
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. ВИМІРЮВАННЯ ЧАСУ СОРТУВАННЯ
# ============================================================

def measure_sorting_times(n=1_000_000, repetitions=30):
    """
    Багаторазово вимірює час сортування n чисел.

    Створення початкових даних не входить у вимірювання.
    Перед кожним сортуванням створюється копія списку.
    """
    rng = np.random.default_rng(42)

    # Генеруємо початкову колекцію випадкових чисел
    original = rng.random(n).tolist()

    times = []

    print(f"Кількість елементів: {n:,}")
    print(f"Кількість вимірювань: {repetitions}")
    print("Вимірюємо час сортування...")

    for i in range(repetitions):
        # Копіювання не враховуємо в часі сортування
        data = original.copy()

        start = time.perf_counter()
        data.sort()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        print(f"Вимірювання {i + 1:2d}: {elapsed:.6f} с")

    return np.array(times)


# ============================================================
# 2. СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ
# ============================================================

def print_statistics(data, title):
    """Виводить основні статистичні характеристики."""
    print(f"\n{title}")
    print("-" * 45)
    print(f"Кількість значень: {len(data)}")
    print(f"Мінімум:           {np.min(data):.6f} с")
    print(f"Максимум:          {np.max(data):.6f} с")
    print(f"Середнє:           {np.mean(data):.6f} с")
    print(f"Медіана:           {np.median(data):.6f} с")
    print(f"Дисперсія:         {np.var(data, ddof=1):.8f}")
    print(f"Стандартне відхилення: {np.std(data, ddof=1):.6f} с")


# ============================================================
# 3. ГІСТОГРАМА ВИМІРЯНИХ ЧАСІВ
# ============================================================

def plot_histogram(data, title):
    """Будує гістограму виміряних значень."""
    plt.figure(figsize=(9, 5))

    plt.hist(
        data,
        bins="auto",
        edgecolor="black",
        alpha=0.75
    )

    plt.title(title)
    plt.xlabel("Час сортування, секунди")
    plt.ylabel("Кількість вимірювань")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. ІДЕНТИФІКАЦІЯ ЗАКОНУ РОЗПОДІЛУ
# ============================================================

def identify_distribution(data):
    """
    Порівнює кілька можливих розподілів:
    нормальний, експоненціальний, логнормальний,
    гамма-розподіл та рівномірний.

    Для порівняння використовує AIC.
    Менше AIC означає кращий компроміс між
    відповідністю даним і складністю моделі.
    """
    candidates = {
        "Нормальний": stats.norm,
        "Експоненціальний": stats.expon,
        "Логнормальний": stats.lognorm,
        "Гамма": stats.gamma,
        "Рівномірний": stats.uniform
    }

    results = []

    for name, distribution in candidates.items():
        try:
            # Часи додатні, тому для цих розподілів
            # фіксуємо початок у нулі
            if name in ["Експоненціальний", "Логнормальний", "Гамма"]:
                params = distribution.fit(data, floc=0)
            else:
                params = distribution.fit(data)

            log_likelihood = np.sum(
                distribution.logpdf(data, *params)
            )

            # AIC = 2k - 2 ln(L)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood

            # Додаткова діагностика: критерій Колмогорова — Смирнова.
            # Зауваження: p-value тут орієнтовне, оскільки
            # параметри розподілу оцінено за тією самою вибіркою.
            ks_statistic, ks_pvalue = stats.kstest(
                data,
                distribution.cdf,
                args=params
            )

            results.append({
                "name": name,
                "distribution": distribution,
                "params": params,
                "aic": aic,
                "ks": ks_statistic,
                "pvalue": ks_pvalue
            })

        except (ValueError, FloatingPointError):
            print(f"Не вдалося підібрати розподіл: {name}")

    if not results:
        raise RuntimeError("Не вдалося підібрати жодного розподілу.")

    results.sort(key=lambda result: result["aic"])

    print("\nПорівняння розподілів:")
    print(
        f"{'Розподіл':<20}"
        f"{'AIC':>14}"
        f"{'KS':>12}"
        f"{'p-value':>12}"
    )

    for result in results:
        print(
            f"{result['name']:<20}"
            f"{result['aic']:>14.3f}"
            f"{result['ks']:>12.4f}"
            f"{result['pvalue']:>12.4f}"
        )

    best = results[0]

    print(f"\nНайкращий за AIC кандидат: {best['name']}")
    print("Параметри:", best["params"])

    return best, results


# ============================================================
# 5. ГЕНЕРАТОР ВИПАДКОВИХ ЗАТРИМОК
# ============================================================

def generate_delays(best_model, count=10_000, seed=123):
    """
    Генерує випадкові затримки за підібраним розподілом.
    """
    rng = np.random.default_rng(seed)

    distribution = best_model["distribution"]
    params = best_model["params"]

    simulated = distribution.rvs(
        *params,
        size=count,
        random_state=rng
    )

    # Час не може бути від'ємним
    simulated = np.maximum(simulated, 0)

    return simulated


# ============================================================
# 6. ПОРІВНЯННЯ РЕАЛЬНИХ І ЗГЕНЕРОВАНИХ ЧАСІВ
# ============================================================

def compare_times(real_times, simulated_times):
    """Порівнює статистики та гістограми двох вибірок."""
    print_statistics(real_times, "РЕАЛЬНІ ЧАСИ СОРТУВАННЯ")
    print_statistics(simulated_times, "ЗГЕНЕРОВАНІ ЧАСИ")

    plt.figure(figsize=(9, 5))

    plt.hist(
        real_times,
        bins="auto",
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Виміряні часи"
    )

    plt.hist(
        simulated_times,
        bins="auto",
        density=True,
        alpha=0.5,
        label="Згенеровані часи"
    )

    plt.title("Порівняння реальних та згенерованих затримок")
    plt.xlabel("Час, секунди")
    plt.ylabel("Щільність")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nПорівняння середніх:")
    print(f"Реальне середнє:     {np.mean(real_times):.6f} с")
    print(f"Згенероване середнє: {np.mean(simulated_times):.6f} с")

    print("\nПорівняння стандартних відхилень:")
    print(f"Реальне:     {np.std(real_times, ddof=1):.6f} с")
    print(f"Згенероване: {np.std(simulated_times, ddof=1):.6f} с")


# ============================================================
# 7. ГОЛОВНА ПРОГРАМА
# ============================================================

def main():
    # Крок 1. Вимірюємо час сортування
    real_times = measure_sorting_times(
        n=1_000_000,
        repetitions=30
    )

    # Крок 2. Аналізуємо реальні вимірювання
    print_statistics(real_times, "РЕЗУЛЬТАТИ ВИМІРЮВАНЬ")
    plot_histogram(real_times, "Гістограма часу сортування")

    # Крок 3. Визначаємо найбільш відповідний розподіл
    best_model, all_models = identify_distribution(real_times)

    # Крок 4. Створюємо генератор затримок
    simulated_times = generate_delays(
        best_model,
        count=10_000
    )

    # Крок 5. Порівнюємо результати
    compare_times(real_times, simulated_times)

    # Крок 6. Формуємо короткий висновок
    mean_difference = abs(
        np.mean(real_times) - np.mean(simulated_times)
    )

    print("\nВИСНОВОК")
    print("-" * 45)
    print(f"Обраний розподіл: {best_model['name']}")
    print(f"Кількість реальних вимірювань: {len(real_times)}")
    print(f"Кількість згенерованих затримок: {len(simulated_times)}")
    print(f"Різниця середніх: {mean_difference:.6f} с")
    print(
        "Точність імітації потрібно оцінювати за гістограмами, "
        "середніми значеннями, розкидом і додатковими перевірками."
    )


if __name__ == "__main__":
    main()
