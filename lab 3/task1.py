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
        return time_min + random.random() * (time_max - time_min)

    @staticmethod
    def norm(time_mean, time_deviation):
        while True:
            t = time_mean + time_deviation * random.gauss(0.0, 1.0)
            if t >= 0:
                return t

    @staticmethod
    def erlang(time_mean, k):
        return sum(FunRand.exp(time_mean / k) for _ in range(k))


class Item:
    next_id = 0

    def __init__(self, t_create, item_type = 1):
        self.id = Item.next_id
        Item.next_id += 1
        self.t_create = t_create 
        self.type = item_type        
        self.orig_type = item_type   


class Route:
    def __init__(self, element, prob = 1.0, priority = 0, block = None):
        self.element = element  #наступний елемент
        self.prob = prob  #ймовірність
        self.priority = priority  
        self.block = block      

    def is_blocked(self, item):
        return self.block is not None and self.block(item)


class Element:
    next_id = 0

    def __init__(self, delay_mean = 1.0, name = None, distribution = "exp", delay_dev = 0.0):
        self.id = Element.next_id
        Element.next_id += 1
        self.name = name if name else f"element{self.id}"
        self.delay_mean = delay_mean
        self.delay_dev = delay_dev
        self.distribution = distribution  
        self.delay_func = None           
        self.tnext = math.inf
        self.tcurr = 0.0
        self.quantity = 0
        self.routes = []                 
        self.route_failure = 0          

    def get_delay(self, item = None):
        if self.delay_func is not None:
            return self.delay_func(item)
        if self.distribution == "exp":
            return FunRand.exp(self.delay_mean)
        if self.distribution == "unif":
            return FunRand.unif(self.delay_mean, self.delay_dev) #delay_mean = min, delay_dev = max
        if self.distribution == "norm":
            return FunRand.norm(self.delay_mean, self.delay_dev)
        if self.distribution == "erlang":
            return FunRand.erlang(self.delay_mean, int(self.delay_dev)) # delay_dev = k
        return self.delay_mean

    def add_route(self, element, prob = 1.0, priority = 0, block = None):
        self.routes.append(Route(element, prob, priority, block))
        return self

    def choose_next(self, item):
        available = [r for r in self.routes if not r.is_blocked(item)]
        if not available:
            return False, None                      
        best = max(r.priority for r in available)
        group = [r for r in available if r.priority == best]
        if len(group) == 1:
            return True, group[0].element
        total = sum(r.prob for r in group)
        x = random.random() * total
        acc = 0.0
        for r in group:
            acc += r.prob
            if x < acc:
                return True, r.element
        return True, group[-1].element

    def send(self, item):
        if not self.routes:
            return                                   
        found, nxt = self.choose_next(item)
        if not found:
            self.route_failure += 1
        elif nxt is not None:
            nxt.in_act(item)

    def in_act(self, item):
        pass

    def out_act(self):
        self.quantity += 1

    def do_statistics(self, delta):
        pass

    def result(self):
        return {"кількість": self.quantity}


class Create(Element):
    def __init__(self, delay_mean, name = "CREATE", distribution = "exp", first_time = 0.0, type_probs = None):
        super().__init__(delay_mean, name, distribution)
        self.tnext = first_time         
        self.type_probs = type_probs     

    def new_type(self):
        if not self.type_probs:
            return 1
        x = random.random()
        acc = 0.0
        for t, p in self.type_probs.items():
            acc += p
            if x < acc:
                return t
        return t

    def out_act(self):
        super().out_act()
        item = Item(self.tcurr, self.new_type())
        self.tnext = self.tcurr + self.get_delay(item)
        self.send(item)


#черга + один або кілька пристроїв
class Process(Element):
    def __init__(self, delay_mean, name = "PROCESS", distribution = "exp", delay_dev = 0.0,
                 channels = 1, max_queue = math.inf, discipline = "FIFO", queue_priority = None):
        super().__init__(delay_mean, name, distribution, delay_dev)
        self.channels = channels        
        self.busy = []                   
        self.queue = []                  
        self.max_queue = max_queue
        self.discipline = discipline # FIFO або LIFO
        self.queue_priority = queue_priority 
        self.on_exit = None              

        self.arrived = 0
        self.failure = 0
        self.mean_queue = 0.0             
        self.mean_load = 0.0             
        self.max_queue_seen = 0
        self.last_arrival = None         
        self.sum_arrival_intervals = 0.0
        self.first_out = None           
        self.last_out = None

    #допоміжні методи
    @property
    def state(self):
        return len(self.busy)

    def _update_tnext(self):
        self.tnext = min((b[0] for b in self.busy), default = math.inf)

    def _take_from_queue(self):
        if self.queue_priority is not None:
            best = min(range(len(self.queue)), key = lambda i: self.queue_priority(self.queue[i]))
            return self.queue.pop(best)
        if self.discipline == "LIFO":
            return self.queue.pop()
        return self.queue.pop(0)

    def start_service(self, item, delay = None):
        if delay is None:
            delay = self.get_delay(item)
        self.busy.append([self.tcurr + delay, item])
        self._update_tnext()

    def try_start(self):
        while self.state < self.channels and self.queue:
            self.start_service(self._take_from_queue())

    def in_act(self, item):
        self.arrived += 1
        if self.last_arrival is not None:
            self.sum_arrival_intervals += self.tcurr - self.last_arrival
        self.last_arrival = self.tcurr

        if self.state < self.channels:              
            self.start_service(item)
        elif len(self.queue) < self.max_queue:       
            self.queue.append(item)
            self.max_queue_seen = max(self.max_queue_seen, len(self.queue))
        else:                                         
            self.failure += 1

    def out_act(self):
        finished = [b for b in self.busy if b[0] == self.tcurr]
        for b in finished:
            self.busy.remove(b)
            item = b[1]
            super().out_act()
            if self.first_out is None:
                self.first_out = self.tcurr
            self.last_out = self.tcurr
            if self.on_exit is not None:
                self.on_exit(item)
            self.try_start()                           
            self.send(item)                            
        self._update_tnext()

    def do_statistics(self, delta):
        self.mean_queue += len(self.queue) * delta
        self.mean_load += self.state * delta

    def result(self):
        res = {
            "надійшло": self.arrived,
            "обслужено": self.quantity,
            "відмов": self.failure + self.route_failure,
            "середня довжина черги": self.mean_queue / self.tcurr,
            "середнє завантаження (зайнятих пристроїв)": self.mean_load / self.tcurr,
        }
        if self.arrived > 0:
            res["ймовірність відмови"] = (self.failure + self.route_failure) / self.arrived
        return res


class Dispose(Element):
    def __init__(self, name = "DISPOSE"):
        super().__init__(0.0, name)
        self.tnext = math.inf
        self.times = {}           
        self.first_out = None
        self.last_out = None

    def in_act(self, item):
        self.quantity += 1
        self.times.setdefault(item.orig_type, []).append(self.tcurr - item.t_create)
        if self.first_out is None:
            self.first_out = self.tcurr
        self.last_out = self.tcurr

    def mean_time(self, item_type=None):
        if item_type is None:
            all_t = [t for lst in self.times.values() for t in lst]
        else:
            all_t = self.times.get(item_type, [])
        return sum(all_t) / len(all_t) if all_t else 0.0

    def result(self):
        return {"вийшло": self.quantity, "середній час перебування": self.mean_time()}


class Model:
    def __init__(self, elements, hooks=None):
        self.elements = elements
        self.hooks = hooks or []   
        self.tcurr = 0.0
        self.tnext = 0.0

    def simulate(self, time, verbose = False):
        while self.tcurr < time:
            #найближча подія
            next_el = min(self.elements, key = lambda e: e.tnext)
            self.tnext = next_el.tnext
            if self.tnext == math.inf:
                break

            #інтервал
            for e in self.elements:
                e.do_statistics(self.tnext - self.tcurr)

            #просування часу
            self.tcurr = self.tnext
            for e in self.elements:
                e.tcurr = self.tcurr
            
            next_el.out_act()
            for e in self.elements:
                if e.tnext == self.tcurr:
                    e.out_act()
            
            for h in self.hooks:
                h(self)
            if verbose:
                self.print_info()

    def print_info(self):
        print(f"t = {self.tcurr:.4f}: " + ", ".join(
            f"{e.name}[зайнято={e.state}, черга={len(e.queue)}]"
            for e in self.elements if isinstance(e, Process)))

    def print_result(self):
        print("\n-------------РЕЗУЛЬТАТИ-------------")
        for e in self.elements:
            print(f"{e.name}:")
            for k, v in e.result().items():
                print(f"    {k:42s} {v:.4f}" if isinstance(v, float) else f"    {k:42s} {v}")


def build_test_network():
    c = Create(2.0, "CREATE")
    p1 = Process(0.6, "СМО1", channels = 1)
    p2 = Process(0.3, "СМО2", channels = 1)
    p3 = Process(0.4, "СМО3", channels = 1)
    p4 = Process(0.1, "СМО4", channels = 2)
    out = Dispose("ВИХІД")
    c.add_route(p1)
    #розгалуження за ймовірностями (усі маршрути з однаковим пріоритетом)
    p1.add_route(p2, prob = 0.15).add_route(p3, prob = 0.13).add_route(p4, prob = 0.30).add_route(out, prob = 0.42)
    p2.add_route(p1)
    p3.add_route(p1)
    p4.add_route(p1)
    return Model([c, p1, p2, p3, p4, out])


def build_priority_demo():
    c = Create(1.0, "CREATE")
    pa = Process(1.5, "СМО_A", channels = 1)
    pb = Process(1.5, "СМО_B", channels = 1)
    c.add_route(pa, priority = 2, block = lambda item: len(pa.queue) >= 3)
    c.add_route(pb, priority = 1)
    return Model([c, pa, pb])


if __name__ == "__main__":
    random.seed(1)
    print("=== Тест 1: мережа з лекції 5 (вибір маршруту за ймовірністю), час 100000 ===")
    model = build_test_network()
    model.simulate(100000.0)
    model.print_result()

    analytic_q = [1.786, 0.003, 0.004, 0.00001]
    analytic_load = [0.714, 0.054, 0.062, 0.036]
    print("\nПорівняння з аналітичним розрахунком:")
    print(f"{'СМО':6s}{'черга (імітація)':>18s}{'черга (аналіт.)':>18s}"
          f"{'завант. (імітація)':>20s}{'завант. (аналіт.)':>20s}")
    for k, p in enumerate(model.elements[1:5]):
        r = p.result()
        print(f"{p.name:6s}{r['середня довжина черги']:18.4f}{analytic_q[k]:18.5f}"
              f"{r['середнє завантаження (зайнятих пристроїв)']:20.4f}{analytic_load[k]:20.3f}")

    random.seed(1)
    print("\n\n=== Тест 2: вибір маршруту за пріоритетом з блокуванням, час 10000 ===")
    model = build_priority_demo()
    model.simulate(10000.0)
    model.print_result()
    print("Черга СМО_A ніколи не перевищує 3, надлишок іде в СМО_B (нижчий пріоритет).")