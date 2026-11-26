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


#обслуговування кількома ідентичними пристроями (каналами)
class Process(Element):
    def __init__(self, delay_mean, name = "PROCESSOR", distribution = "exp", max_queue = math.inf, channels = 1):
        super().__init__(delay_mean, name, distribution)
        self.channels = channels
        # для кожного каналу зберігаємо момент виходу вимоги з нього
        self.channel_tnext = [math.inf] * channels # math.inf означає, що канал вільний
        self.tnext = math.inf      
        self.state = 0 #кількість зайнятих каналів
        self.queue = 0
        self.max_queue = max_queue
        self.failure = 0
        self.arrived = 0
        self.mean_queue = 0.0
        self.mean_load = 0.0      

    def _update_tnext(self):
        self.tnext = min(self.channel_tnext)

    def in_act(self):
        self.arrived += 1
        if math.inf in self.channel_tnext:                
            i = self.channel_tnext.index(math.inf)
            self.channel_tnext[i] = self.tcurr + self.get_delay()
            self.state += 1
        elif self.queue < self.max_queue:                 
            self.queue += 1
        else:                                           
            self.failure += 1
        self._update_tnext()

    def out_act(self):
        for i in range(self.channels):
            if self.channel_tnext[i] == self.tcurr:
                super().out_act()                         
                self.channel_tnext[i] = math.inf           
                self.state -= 1
                if self.queue > 0:                         
                    self.queue -= 1
                    self.channel_tnext[i] = self.tcurr + self.get_delay()
                    self.state += 1
                if self.next_element is not None:          
                    self.next_element.in_act()
        self._update_tnext()

    def do_statistics(self, delta):
        self.mean_queue += self.queue * delta
        self.mean_load += self.state * delta

    def print_info(self):
        super().print_info()
        print(f"    queue = {self.queue}, failure = {self.failure}, "
              f"зайнято каналів = {self.state}/{self.channels}")


class Model:
    def __init__(self, elements):
        self.elements = elements
        self.tnext = 0.0
        self.tcurr = 0.0

    def simulate(self, time, verbose = True):
        while self.tcurr < time:
            next_el = min(self.elements, key = lambda e: e.tnext)
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
            res[f"Сер. к-сть зайнятих пристроїв СМО{k}"] = p.mean_load / self.tcurr
        total_fail = sum(p.failure for p in processes)
        res["Ймовірність відмови (вся мережа)"] = total_fail / creator.quantity
        return res

    def print_result(self):
        print("\n-------------РЕЗУЛЬТАТИ-------------")
        for name, value in self.results().items():
            print(f"{name:42s} {value:.4f}" if isinstance(value, float) else f"{name:42s} {value}")


def build_model(params):
    c = Create(params["d_create"])
    p1 = Process(params["d1"], "PROCESS1", max_queue=params["q1"], channels=params["n1"])
    p2 = Process(params["d2"], "PROCESS2", max_queue=params["q2"], channels=params["n2"])
    p3 = Process(params["d3"], "PROCESS3", max_queue=params["q3"], channels=params["n3"])
    c.next_element = p1
    p1.next_element = p2
    p2.next_element = p3
    p3.next_element = None  
    return Model([c, p1, p2, p3])


INPUT_NAMES = {
    "d_create": "Середній інтервал надходження",
    "n1": "Кількість пристроїв СМО1",
    "d1": "Середній час обслуговування СМО1",
    "q1": "Макс. довжина черги СМО1",
    "n2": "Кількість пристроїв СМО2",
    "d2": "Середній час обслуговування СМО2",
    "q2": "Макс. довжина черги СМО2",
    "n3": "Кількість пристроїв СМО3",
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
    width = 42
    header = f"{'Змінна':{width}s}" + "".join(f"{'Прогін ' + str(i + 1):>11s}" for i in range(len(runs)))
    print(header)
    print("-" * len(header))
    print("ВХІДНІ ЗМІННІ")
    for key, title in INPUT_NAMES.items():
        cells = ""
        for params, change, _ in runs:
            mark = "*" if key in change else " "
            cells += f"{fmt(params[key]) + mark:>11s}"
        print(f"{title:{width}s}{cells}")
    print("ВИХІДНІ ЗМІННІ")
    for name in out_names:
        print(f"{name:{width}s}" + "".join(f"{fmt(r[2][name]):>11s}" for r in runs))

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
    base = {"d_create": 1.0, "n1": 2, "d1": 1.5, "q1": 5,
            "n2": 3, "d2": 2.5, "q2": 5, "n3": 1, "d3": 0.8, "q3": 5}

    print("\n\n--1 прогін, 1000--")
    random.seed(1)
    model = build_model(base)
    model.simulate(1000.0, verbose=False)
    model.print_result()

    print("\n\n--Верифікація (10000)--\n")
    changes = [ {}, {"n1": 1}, {"n1": 4}, {"n2": 1}, {"n2": 6},                 
        {"d_create": 0.5}, {"d_create": 3.0},      
        {"d_create": 2.0, "n1": 1, "d1": 1.0, "n2": 1, "d2": 1.5, "n3": 1, "d3": 1.0},
    ]
    verification(base, changes, csv_name="verification_task5.csv")