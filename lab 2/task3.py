import math
import random


class FunRand:
    @staticmethod
    def exp(time_mean):
        a = 0.0
        while a == 0.0:          
            a = random.random()
        return -time_mean * math.log(a)

    @staticmethod
    def unif(time_min, time_max):
        a = 0.0
        while a == 0.0:
            a = random.random()
        return time_min + a * (time_max - time_min)

    @staticmethod
    def norm(time_mean, time_deviation):
        return time_mean + time_deviation * random.gauss(0.0, 1.0)


class Element:
    next_id = 0  

    def __init__(self, delay_mean = 1.0, name = None, distribution = "exp"):
        self.id = Element.next_id
        Element.next_id += 1
        self.name = name if name else f"element{self.id}"

        self.delay_mean = delay_mean      
        self.delay_dev = 0.0           
        self.distribution = distribution  

        self.tnext = 0.0          
        self.tcurr = 0.0         
        self.state = 0            
        self.quantity = 0         
        self.next_element = None 

    def get_delay(self):
        if self.distribution == "exp":
            return FunRand.exp(self.delay_mean)
        if self.distribution == "norm":
            return FunRand.norm(self.delay_mean, self.delay_dev)
        if self.distribution == "unif":
            return FunRand.unif(self.delay_mean, self.delay_dev)
        return self.delay_mean  

    def in_act(self):
        pass

    def out_act(self):
        self.quantity += 1

    def do_statistics(self, delta):
        pass

    def print_info(self):
        print(f"{self.name}: state = {self.state}, quantity = {self.quantity}, "
              f"tnext = {self.tnext:.4f}")

    def print_result(self):
        print(f"{self.name}: quantity = {self.quantity}")


class Create(Element):
    def __init__(self, delay_mean, name = "CREATOR", distribution = "exp"):
        super().__init__(delay_mean, name, distribution)
        self.tnext = 0.0  

    def out_act(self):
        super().out_act()                         
        self.tnext = self.tcurr + self.get_delay()   
        self.next_element.in_act()                     


class Process(Element):
    def __init__(self, delay_mean, name = "PROCESSOR", distribution = "exp", max_queue = math.inf):
        super().__init__(delay_mean, name, distribution)
        self.tnext = math.inf
        self.queue = 0
        self.max_queue = max_queue
        self.failure = 0
        self.arrived = 0         
        self.mean_queue = 0.0   
        self.mean_load = 0.0     

    def in_act(self):
        self.arrived += 1
        if self.state == 0:
            self.state = 1
            self.tnext = self.tcurr + self.get_delay()
        elif self.queue < self.max_queue:
            self.queue += 1
        else:
            self.failure += 1

    def out_act(self):
        super().out_act()
        self.tnext = math.inf
        self.state = 0
        if self.queue > 0:
            self.queue -= 1
            self.state = 1
            self.tnext = self.tcurr + self.get_delay()
        #передаємо обслужену вимогу наступному блоку
        if self.next_element is not None:
            self.next_element.in_act()

    def do_statistics(self, delta):
        self.mean_queue += self.queue * delta
        self.mean_load += self.state * delta

    def print_info(self):
        super().print_info()
        print(f"    queue = {self.queue}, failure = {self.failure}")


class Model:
    def __init__(self, elements):
        self.elements = elements
        self.tnext = 0.0
        self.tcurr = 0.0

    def simulate(self, time, verbose=True):
        while self.tcurr < time:
            next_el = min(self.elements, key=lambda e: e.tnext)
            self.tnext = next_el.tnext
            if verbose:
                print(f"\nПодія в {next_el.name}, час = {self.tnext:.4f}")
            for e in self.elements:
                e.do_statistics(self.tnext - self.tcurr)
            self.tcurr = self.tnext
            for e in self.elements:
                e.tcurr = self.tcurr
            next_el.out_act()
            for e in self.elements:
                if e.tnext == self.tcurr:
                    e.out_act()
            if verbose:
                for e in self.elements:
                    e.print_info()

    def results(self):
        res = {}
        processes = [e for e in self.elements if isinstance(e, Process)]
        creator = self.elements[0]
        res["Кількість вимог, що надійшли"] = creator.quantity
        for k, p in enumerate(processes, start=1):
            res[f"Обслужено в СМО{k}"] = p.quantity
            res[f"Відмов в СМО{k}"] = p.failure
            res[f"Середня довжина черги СМО{k}"] = p.mean_queue / self.tcurr
            res[f"Середнє завантаження СМО{k}"] = p.mean_load / self.tcurr
        total_fail = sum(p.failure for p in processes)
        res["Ймовірність відмови (вся мережа)"] = total_fail / creator.quantity
        return res

    def print_result(self):
        print("\n-------------РЕЗУЛЬТАТИ-------------")
        for name, value in self.results().items():
            print(f"{name:40s} {value:.4f}" if isinstance(value, float) else f"{name:40s} {value}")


#CREATE -> PROCESS1 -> PROCESS2 -> PROCESS3 -> DISPOSE
def build_model(params):
    c = Create(params["d_create"])
    p1 = Process(params["d1"], "PROCESS1", max_queue=params["q1"])
    p2 = Process(params["d2"], "PROCESS2", max_queue=params["q2"])
    p3 = Process(params["d3"], "PROCESS3", max_queue=params["q3"])
    c.next_element = p1
    p1.next_element = p2
    p2.next_element = p3
    p3.next_element = None
    return Model([c, p1, p2, p3])


# Назви вхідних змінних для таблиці
INPUT_NAMES = {
    "d_create": "Середній інтервал надходження",
    "d1": "Середній час обслуговування СМО1",
    "q1": "Макс. довжина черги СМО1",
    "d2": "Середній час обслуговування СМО2",
    "q2": "Макс. довжина черги СМО2",
    "d3": "Середній час обслуговування СМО3",
    "q3": "Макс. довжина черги СМО3",
}


def fmt(v):
    if isinstance(v, float):
        return "∞" if v == math.inf else f"{v:.3f}"
    return str(v)

def verification(base, changes, time = 10000.0, csv_name = None):
    runs = []
    for change in changes:
        params = dict(base)
        params.update(change)          
        random.seed(42)       
        model = build_model(params)
        model.simulate(time, verbose=False)
        runs.append((params, change, model.results()))

    out_names = list(runs[0][2].keys())
    width = 38
    header = f"{'Змінна':{width}s}" + "".join(f"{'Прогін ' + str(i + 1):>12s}" for i in range(len(runs)))
    print(header)
    print("-" * len(header))
    print("ВХІДНІ ЗМІННІ (* - змінене значення)")
    for key, title in INPUT_NAMES.items():
        cells = ""
        for params, change, _ in runs:
            mark = "*" if key in change else " "
            cells += f"{fmt(params[key]) + mark:>12s}"
        print(f"{title:{width}s}{cells}")
    print("ВИХІДНІ ЗМІННІ")
    for name in out_names:
        print(f"{name:{width}s}" + "".join(f"{fmt(r[2][name]):>12s}" for r in runs))

    if csv_name:
        import csv
        with open(csv_name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Змінна"] + [f"Прогін {i + 1}" for i in range(len(runs))])
            for key, title in INPUT_NAMES.items():
                w.writerow([title] + [fmt(r[0][key]) + ("*" if key in r[1] else "") for r in runs])
            for name in out_names:
                w.writerow([name] + [fmt(r[2][name]) for r in runs])
        print(f"\nТаблицю збережено у файл {csv_name}")


if __name__ == "__main__":
    random.seed(1)
    base = {"d_create": 2.0, "d1": 1.0, "q1": 5, "d2": 1.5, "q2": 5, "d3": 1.0, "q3": 5}
    model = build_model(base)
    model.simulate(1000.0, verbose = False)
    print("--Завдання 3: мережа з трьох СМО, 1000--")
    model.print_result()

    print("\n\n--Завдання 4: верифікація моделі (10000)--\n")
    changes = [ {}, {"d_create": 1.0}, {"d_create": 5.0}, {"d1": 1.8}, {"d2": 3.0},          
        {"q2": 1}, {"d3": 0.2}, {"q1": 0, "q2": 0, "q3": 0}]
    verification(base, changes, csv_name="verification_task4.csv")