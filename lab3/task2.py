import random
from task1 import Create, Process, Dispose, Model, Item, FunRand


def build_bank():
    create = Create(0.5, "ПРИБУТТЯ", first_time = 0.1)
    cash1 = Process(0.3, "КАСА 1", channels = 1, max_queue = 3)
    cash2 = Process(0.3, "КАСА 2", channels = 1, max_queue = 3)
    out = Dispose("ВИЇЗД")

    def cars(p):
        return len(p.queue) + p.state
  
    create.add_route(cash1, priority = 2, block = lambda item: cars(cash1) > cars(cash2))
    create.add_route(cash2, priority = 1)
    cash1.add_route(out)
    cash2.add_route(out)

    #початкові умови: обидва касири зайняті (час обслуговування norm(1, 0.3)), у кожній черзі по 2 автомобілі
    for cash in (cash1, cash2):
        cash.start_service(Item(0.0), delay = FunRand.norm(1.0, 0.3))
        cash.queue = [Item(0.0), Item(0.0)]

    switches = {"кількість": 0}

    def lane_change(model):
        q1, q2 = cash1.queue, cash2.queue
        while abs(len(q1) - len(q2)) >= 2:
            if len(q1) > len(q2):
                q2.append(q1.pop())        
            else:
                q1.append(q2.pop())
            switches["кількість"] += 1
        
        cash1.try_start()
        cash2.try_start()

    model = Model([create, cash1, cash2, out], hooks=[lane_change])
    return model, create, cash1, cash2, out, switches


if __name__ == "__main__":
    random.seed(1)
    T = 10000.0
    model, create, cash1, cash2, out, switches = build_bank()
    model.simulate(T)

    print(f"--Банк для автомобілістів, час моделювання {T:.0f}--\n")
    load1 = cash1.mean_load / model.tcurr
    load2 = cash2.mean_load / model.tcurr
    in_bank = (cash1.mean_queue + cash1.mean_load + cash2.mean_queue + cash2.mean_load) / model.tcurr
    dep_interval = (out.last_out - out.first_out) / (out.quantity - 1)
    refused = cash1.failure + cash2.failure

    print(f"1) Середнє завантаження касира 1: {load1:.4f}")
    print(f"   Середнє завантаження касира 2: {load2:.4f}")
    print(f"2) Середнє число клієнтів у банку: {in_bank:.4f}")
    print(f"3) Середній інтервал між від'їздами: {dep_interval:.4f}")
    print(f"4) Середній час перебування клієнта в банку: {out.mean_time():.4f}")
    print(f"5) Середня довжина черги до каси 1: {cash1.mean_queue / model.tcurr:.4f}")
    print(f"   Середня довжина черги до каси 2: {cash2.mean_queue / model.tcurr:.4f}")
    print(f"6) Відсоток клієнтів, яким відмовлено: {refused / create.quantity * 100:.2f} %"
          f"  ({refused} з {create.quantity})")
    print(f"7) Число змін під'їзних смуг: {switches['кількість']}")