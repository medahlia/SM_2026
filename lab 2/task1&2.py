"""
Комп'ютерний практикум 2. Завдання 1 і 2.

1. Алгоритм імітації простої моделі обслуговування одним пристроєм
   (CREATE -> PROCESS -> DISPOSE) з використанням ООП.
2. Модифікація: обчислення середнього завантаження пристрою.

Структура класів така сама, як у лекції 4 (Java):
FunRand, Element, Create, Process, Model.
"""

import math
import random


# ---------------------------------------------------------------------------
# Генератори випадкових величин
# ---------------------------------------------------------------------------
class FunRand:
    @staticmethod
    def exp(time_mean):
        """Експоненційний розподіл із середнім time_mean."""
        a = 0.0
        while a == 0.0:          # log(0) не можна, тому відкидаємо нуль
            a = random.random()
        return -time_mean * math.log(a)

    @staticmethod
    def unif(time_min, time_max):
        """Рівномірний розподіл на [time_min, time_max]."""
        a = 0.0
        while a == 0.0:
            a = random.random()
        return time_min + a * (time_max - time_min)

    @staticmethod
    def norm(time_mean, time_deviation):
        """Нормальний розподіл (Гаусса)."""
        return time_mean + time_deviation * random.gauss(0.0, 1.0)


# ---------------------------------------------------------------------------
# Базовий елемент моделі
# ---------------------------------------------------------------------------
class Element:
    next_id = 0  # лічильник для унікальних номерів елементів

    def __init__(self, delay_mean=1.0, name=None, distribution="exp"):
        self.id = Element.next_id
        Element.next_id += 1
        self.name = name if name else f"element{self.id}"

        self.delay_mean = delay_mean      # середня часова затримка
        self.delay_dev = 0.0              # відхилення (для norm / unif)
        self.distribution = distribution  # "exp", "norm", "unif" або "" (детермінована)

        self.tnext = 0.0          # момент наступної події елемента
        self.tcurr = 0.0          # поточний модельний час
        self.state = 0            # 0 - вільний, 1 - зайнятий
        self.quantity = 0         # кількість об'єктів, що вийшли з елемента
        self.next_element = None  # наступний елемент у маршруті

    def get_delay(self):
        """Генерує часову затримку за заданим законом розподілу."""
        if self.distribution == "exp":
            return FunRand.exp(self.delay_mean)
        if self.distribution == "norm":
            return FunRand.norm(self.delay_mean, self.delay_dev)
        if self.distribution == "unif":
            return FunRand.unif(self.delay_mean, self.delay_dev)
        return self.delay_mean  # детермінована затримка

    def in_act(self):
        """Вхід об'єкта в елемент (перевизначається в нащадках)."""
        pass

    def out_act(self):
        """Вихід об'єкта з елемента."""
        self.quantity += 1

    def do_statistics(self, delta):
        """Накопичення статистики за інтервал delta (перевизначається)."""
        pass

    def print_info(self):
        print(f"{self.name}: state = {self.state}, quantity = {self.quantity}, "
              f"tnext = {self.tnext:.4f}")

    def print_result(self):
        print(f"{self.name}: quantity = {self.quantity}")


# ---------------------------------------------------------------------------
# CREATE - генератор вимог
# ---------------------------------------------------------------------------
class Create(Element):
    def __init__(self, delay_mean, name="CREATOR", distribution="exp"):
        super().__init__(delay_mean, name, distribution)
        self.tnext = 0.0  # імітація починається з події Create

    def out_act(self):
        super().out_act()                              # +1 створена вимога
        self.tnext = self.tcurr + self.get_delay()     # момент наступного надходження
        self.next_element.in_act()                     # передаємо вимогу далі


# ---------------------------------------------------------------------------
# PROCESS - пристрій обслуговування з чергою
# ---------------------------------------------------------------------------
class Process(Element):
    def __init__(self, delay_mean, name="PROCESSOR", distribution="exp", max_queue=math.inf):
        super().__init__(delay_mean, name, distribution)
        self.tnext = math.inf      # поки пристрій вільний, подій виходу не очікується
        self.queue = 0             # поточна довжина черги
        self.max_queue = max_queue # обмеження на довжину черги
        self.failure = 0           # кількість відмов
        self.mean_queue = 0.0      # інтеграл довжини черги за часом
        self.mean_load = 0.0       # ЗАВДАННЯ 2: інтеграл зайнятості пристрою за часом

    def in_act(self):
        if self.state == 0:                            # пристрій вільний -> займаємо
            self.state = 1
            self.tnext = self.tcurr + self.get_delay()
        elif self.queue < self.max_queue:              # є місце в черзі
            self.queue += 1
        else:                                          # черга заповнена -> відмова
            self.failure += 1

    def out_act(self):
        super().out_act()                              # +1 обслужена вимога
        self.tnext = math.inf
        self.state = 0
        if self.queue > 0:                             # беремо наступну вимогу з черги
            self.queue -= 1
            self.state = 1
            self.tnext = self.tcurr + self.get_delay()

    def do_statistics(self, delta):
        # Площа під графіком довжини черги та під графіком зайнятості
        self.mean_queue += self.queue * delta
        self.mean_load += self.state * delta           # ЗАВДАННЯ 2

    def print_info(self):
        super().print_info()
        print(f"    queue = {self.queue}, failure = {self.failure}")


# ---------------------------------------------------------------------------
# Модель: просування часу за принципом найближчої події
# ---------------------------------------------------------------------------
class Model:
    def __init__(self, elements):
        self.elements = elements
        self.tnext = 0.0
        self.tcurr = 0.0

    def simulate(self, time, verbose=True):
        while self.tcurr < time:
            # 1) шукаємо момент найближчої події серед усіх елементів
            next_el = min(self.elements, key=lambda e: e.tnext)
            self.tnext = next_el.tnext
            if verbose:
                print(f"\nПодія в {next_el.name}, час = {self.tnext:.4f}")

            # 2) збираємо статистику за інтервал [tcurr, tnext]
            for e in self.elements:
                e.do_statistics(self.tnext - self.tcurr)

            # 3) просуваємо час
            self.tcurr = self.tnext
            for e in self.elements:
                e.tcurr = self.tcurr

            # 4) виконуємо подію (і всі інші, що припали на цей самий момент)
            next_el.out_act()
            for e in self.elements:
                if e.tnext == self.tcurr:
                    e.out_act()

            if verbose:
                for e in self.elements:
                    e.print_info()
        if verbose:
            self.print_result()

    def print_result(self):
        print("\n-------------РЕЗУЛЬТАТИ-------------")
        for e in self.elements:
            e.print_result()
            if isinstance(e, Process):
                arrived = e.quantity + e.failure + e.queue + e.state
                print(f"    середня довжина черги = {e.mean_queue / self.tcurr:.4f}")
                print(f"    ймовірність відмови   = {e.failure / arrived:.4f}")
                print(f"    середнє завантаження  = {e.mean_load / self.tcurr:.4f}")  # ЗАВДАННЯ 2


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(1)

    # Короткий прогін із протоколом подій, щоб було видно роботу алгоритму
    print("=== Протокол перших подій (час моделювання 10) ===")
    c = Create(2.0)
    p = Process(1.0, max_queue=5)
    c.next_element = p
    Model([c, p]).simulate(10.0, verbose=True)

    # Основний прогін
    print("\n\n=== Основний прогін (час моделювання 1000) ===")
    c = Create(2.0)
    p = Process(1.0, max_queue=5)
    c.next_element = p
    model = Model([c, p])
    model.simulate(1000.0, verbose=False)
    model.print_result()
    # Перевірка: для M/M/1 завантаження ≈ ρ = 1/2 = 0.5
    print("\nТеоретичне завантаження ρ = delayProcess / delayCreate = 1.0 / 2.0 = 0.5")