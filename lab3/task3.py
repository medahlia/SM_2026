import math
import random
from task1 import Create, Process, Dispose, Model, FunRand

REG_TIME = {1: 15.0, 2: 40.0, 3: 30.0} # середній час реєстрації за типом


def build_hospital():
    create = Create(15.0, "НАДХОДЖЕННЯ", type_probs = {1: 0.5, 2: 0.1, 3: 0.4})

    reception = Process(0, "ПРИЙМАЛЬНЕ", channels = 2,
                        queue_priority = lambda item: 0 if item.type == 1 else 1)
    reception.delay_func = lambda item: FunRand.exp(REG_TIME[item.type])

    escort = Process(3, "СУПРОВІДНІ", distribution = "unif", delay_dev = 8, channels = 3)
    to_lab = Process(2, "ПЕРЕХІД У ЛАБ.", distribution = "unif", delay_dev = 5, channels = math.inf)
    lab_reg = Process(4.5, "РЕЄСТРАТУРА ЛАБ.", distribution = "erlang", delay_dev = 3, channels = 1)
    lab = Process(4.0, "ЛАБОРАНТИ", distribution = "erlang", delay_dev = 2, channels = 2)
    to_reception = Process(2, "ПЕРЕХІД У ПРИЙМ.", distribution = "unif", delay_dev = 5, channels = math.inf)
    ward = Dispose("ПАЛАТА")
    lab_exit = Dispose("ВИХІД З ЛАБ.")

    create.add_route(reception)
    reception.add_route(escort, block = lambda item: item.type != 1)   
    reception.add_route(to_lab, block = lambda item: item.type == 1)   
    escort.add_route(ward)
    to_lab.add_route(lab_reg)
    lab_reg.add_route(lab)
    lab.add_route(to_reception, block = lambda item: item.type != 2)   
    lab.add_route(lab_exit, block = lambda item: item.type != 3)       
    to_reception.add_route(reception)

    def become_type1(item):
        item.type = 1        
    to_reception.on_exit = become_type1

    elements = [create, reception, escort, to_lab, lab_reg, lab, to_reception, ward, lab_exit]
    return Model(elements), lab_reg, ward, lab_exit


if __name__ == "__main__":
    random.seed(1)
    T = 100000.0   
    model, lab_reg, ward, lab_exit = build_hospital()
    model.simulate(T)

    print(f"--Лікарня, {T:.0f}--")
    model.print_result()

    print("\n")
    print(f"Середній час у системі, хворі типу 1 (до палати): {ward.mean_time(1):.2f} хв")
    print(f"Середній час у системі, хворі типу 2 (до палати): {ward.mean_time(2):.2f} хв")
    print(f"Середній час у системі, хворі типу 3 (вихід з лаб.): {lab_exit.mean_time(3):.2f} хв")
    print(f"Середній інтервал між прибуттями в лабораторію: "
          f"{lab_reg.sum_arrival_intervals / (lab_reg.arrived - 1):.2f} хв")