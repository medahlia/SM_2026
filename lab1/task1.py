import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


# ЗАГАЛЬНІ ФУНКЦІЇ
#Обчислює середнє значення та дисперсію.
def calculate_statistics(x):
    mean = np.mean(x)
    variance = np.var(x, ddof=1)   # незміщена оцінка
    return mean, variance

#Будує гістограму частот
def plot_histogram(x, title, bins=20):
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Значення випадкової величини")
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

#Обчислює критерій Пірсона за формулою: chi^2 = sum((O_i - E_i)^2 / E_i)
def chi_square_test(observed, expected, alpha=0.05, df=None):
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if np.any(expected <= 0):
        raise ValueError("Очікувані частоти мають бути додатними.")

    if np.any(expected < 5):
        print(
            "УВАГА: очікувана частота в одному з інтервалів "
            "менша за 5. Інтервали слід об'єднати."
        )

    chi_square_value = np.sum(
        (observed - expected) ** 2 / expected
    )

    if df is None:
        df = len(observed) - 1

    critical_value = chi2.ppf(1 - alpha, df)

    print(f"Обчислене значення χ²: {chi_square_value:.4f}")
    print(f"Критичне значення χ²: {critical_value:.4f}")
    print(f"Ступені свободи: {df}")
    print(f"Рівень значущості: {alpha}")

    if chi_square_value <= critical_value:
        print("гіпотезу НЕ відхиляємо")
        accepted = True
    else:
        print("гіпотезу відхиляємо")
        accepted = False

    return chi_square_value, critical_value, accepted


# СПОСІБ 1. ЕКСПОНЕНЦІАЛЬНИЙ РОЗПОДІЛ
# Генерує n чисел за формулою: x_i = -(1 / lambda) * ln(xi_i), де xi_i рівномірно розподілені на (0; 1).
def generate_exponential(n, lam):
    xi = 1.0 - np.random.random(n)    
    x = -(1 / lam) * np.log(xi)
    return x

#Перевіряє гіпотезу про експоненціальний розподіл
def test_exponential_distribution(x, lam, bins=10, alpha=0.05, n_estimated=0):
    n = len(x)

    probabilities = np.arange(1, bins) / bins
    inner_edges = -np.log(1 - probabilities) / lam

    edges = np.concatenate(([0.0], inner_edges, [np.inf]))

    observed, _ = np.histogram(x, bins=edges)

    expected = np.full(bins, n / bins)

    print("\nФактичні та очікувані частоти:")
    print(f"{'Інтервал':>20} {'Фактична':>12} {'Очікувана':>12}")

    for i in range(bins):
        if i == bins - 1:
            interval = f"[{edges[i]:.3f}; ∞)"
        else:
            interval = f"[{edges[i]:.3f}; {edges[i+1]:.3f})"

        print(
            f"{interval:>20} "
            f"{observed[i]:12d} "
            f"{expected[i]:12.2f}"
        )

    return chi_square_test(
        observed, expected, alpha, df=bins - 1 - n_estimated
    )


def run_exponential_task(n=10_000):
    print("\n" + "=" * 60)
    print("СПОСІБ 1. ЕКСПОНЕНЦІАЛЬНИЙ РОЗПОДІЛ")
    print("=" * 60)

    lambda_values = [0.5, 1.0, 2.0, 5.0]

    for lam in lambda_values:
        x = generate_exponential(n, lam)

        mean, variance = calculate_statistics(x)

        print("\n" + "-" * 50)
        print(f"Параметр λ = {lam}")
        print(f"Кількість чисел: {n}")
        print(f"Вибіркове середнє: {mean:.4f}")
        print(f"Теоретичне середнє: {1 / lam:.4f}")
        print(f"Вибіркова дисперсія: {variance:.4f}")
        print(f"Теоретична дисперсія: {1 / lam**2:.4f}")

        plot_histogram(
            x,
            f"Експоненціальний розподіл, λ = {lam}"
        )

        print("\nПеревірка гіпотези H0:")
        print(f"H0: дані мають експоненціальний розподіл λ = {lam}")

        test_exponential_distribution(x, lam)

        # Формування гіпотези за даними: для експоненційного закону
        # mu = sigma = 1/lambda, тобто lambda оцінюється як 1/mean 
        lam_hat = 1 / mean
        print(
            f"\nГіпотеза за даними: σ/μ = {np.sqrt(variance) / mean:.3f} "
            f"(для експоненційного ≈ 1), λ̂ = 1/μ = {lam_hat:.4f}"
        )
        print("Перевірка з оціненим λ̂ (df зменшується на 1):")
        test_exponential_distribution(x, lam_hat, n_estimated=1)


# СПОСІБ 2. РЕКУРЕНТНИЙ ГЕНЕРАТОР
# Генерує числа за формулами: z_(i+1) = a * z_i mod c, x_(i+1) = z_(i+1) / c
def generate_congruential(n, a, c, z0=1):
    z = z0
    numbers = []

    for _ in range(n):
        z = (a * z) % c
        x = z / c
        numbers.append(x)

    return np.array(numbers)

#Перевіряє гіпотезу про рівномірний розподіл
def test_uniform_distribution(x, bins=20, alpha=0.05):
    n = len(x)

    edges = np.linspace(0, 1, bins + 1)

    observed, _ = np.histogram(x, bins=edges)

    expected = np.full(bins, n / bins)

    print("\nФактичні та очікувані частоти:")
    print(f"{'Інтервал':>18} {'Фактична':>12} {'Очікувана':>12}")

    for i in range(bins):
        print(
            f"[{edges[i]:.2f}; {edges[i+1]:.2f})".rjust(18)
            + f"{observed[i]:12d}"
            f"{expected[i]:12.2f}"
        )

    return chi_square_test(
        observed, expected, alpha, df=bins - 1
    )


def run_congruential_task(n=10_000):
    print("\n" + "=" * 60)
    print("СПОСІБ 2. РЕКУРЕНТНИЙ ГЕНЕРАТОР")
    print("=" * 60)

    # Основні параметри з методички:
    # a = 5^13, c = 2^31
    parameter_sets = [
        (5**13, 2**31),
        (5**15, 2**31),
        (16_807, 2**31 - 1),
        (1_103_515_245, 2**31)
    ]

    for a, c in parameter_sets:
        x = generate_congruential(n, a, c, z0=1)

        mean, variance = calculate_statistics(x)

        print("\n" + "-" * 50)
        print(f"Параметр a = {a}")
        print(f"Параметр c = {c}")
        print(f"Початкове значення z0 = 1")
        print(f"Кількість чисел: {n}")
        print(f"Мінімальне значення: {np.min(x):.6f}")
        print(f"Максимальне значення: {np.max(x):.6f}")
        print(f"Вибіркове середнє: {mean:.6f}")
        print(f"Теоретичне середнє: {0.5:.6f}")
        print(f"Вибіркова дисперсія: {variance:.6f}")
        print(f"Теоретична дисперсія: {1 / 12:.6f}")
        print(f"Кількість різних значень: {len(np.unique(x))}")

        plot_histogram(
            x,
            f"Рекурентний генератор: a = {a}, c = {c}"
        )

        print("\nПеревірка гіпотези H0:")
        print("H0: числа рівномірно розподілені на (0; 1)")

        test_uniform_distribution(x)


# ============================================================
def main():
    np.random.seed(42)

    n = 10_000

    run_exponential_task(n)
    run_congruential_task(n)

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()