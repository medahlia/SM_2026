import math
import random


#генератори випадкових величин
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
        self.distribution = distribution # distribution = розподіл

        self.tnext = 0.0          
        self.tcurr = 0.0          
        self.state = 0 # 0 вільний, 1 зайнятий
        self.quantity = 0         
        self.next_element = None 

    def get_delay(self):
        if self.distribution == "exp":
            return FunRand.exp(self.delay_mean)
        if self.distribution == "norm":
            return FunRand.norm(self.delay_mean, self.delay_dev)
        if self.distribution == "unif":
            return FunRand.unif(self.delay_mean, self.delay_dev)
        return self.delay_mean  # детермінована 

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
        self.tnext = 0.0  # імітація починається з події Create

    def out_act(self):
        super().out_act()                              
        self.tnext = self.tcurr + self.get_delay()    
        self.next_element.in_act()                   


class Process(Element):
    def __init__(self, delay_mean, name = "PROCESSOR", distribution = "exp", max_queue = math.inf):
        super().__init__(delay_mean, name, distribution)
        self.tnext = math.inf    
        self.queue = 0 # поточна
        self.max_queue = max_queue 
        self.failure = 0 # к-сть
        self.mean_queue = 0.0 # інтеграл довжини черги за часом
        self.mean_load = 0.0 # інтеграл зайнятості пристрою за часом

    def in_act(self):
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

    def do_statistics(self, delta):
        self.mean_queue += self.queue * delta
        self.mean_load += self.state * delta        

    def print_info(self):
        super().print_info()
        print(f"    queue = {self.queue}, failure = {self.failure}")


# просування часу за принципом найближчої події
class Model:
    def __init__(self, elements):
        self.elements = elements
        self.tnext = 0.0
        self.tcurr = 0.0

    def simulate(self, time, verbose = True):
        while self.tcurr < time:
            #шукаємо момент найближчої події серед усіх елементів
            next_el = min(self.elements, key=lambda e: e.tnext)
            self.tnext = next_el.tnext
            if verbose:
                print(f"\nПодія в {next_el.name}, час = {self.tnext:.4f}")

            #збираємо статистику за інтервал [tcurr, tnext]
            for e in self.elements:
                e.do_statistics(self.tnext - self.tcurr)

            #просуваємо час
            self.tcurr = self.tnext
            for e in self.elements:
                e.tcurr = self.tcurr

            #виконуємо подію (і всі інші, що припали на цей самий момент)
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
                print(f"    середнє завантаження  = {e.mean_load / self.tcurr:.4f}") 


#----------------------------
if __name__ == "__main__":
    random.seed(1)

    print("--Протокол перших подій (час моделювання 10)--")
    c = Create(2.0)
    p = Process(1.0, max_queue = 5)
    c.next_element = p
    Model([c, p]).simulate(10.0, verbose = True)

    # основний прогін
    print("\n\n--Основний прогін (час моделювання 1000)--")
    c = Create(2.0)
    p = Process(1.0, max_queue = 5)
    c.next_element = p
    model = Model([c, p])
    model.simulate(1000.0, verbose = False)
    model.print_result()
    print("\nТеоретичне завантаження ρ = delayProcess / delayCreate = 1.0 / 2.0 = 0.5")