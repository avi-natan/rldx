import copy
from abc import ABC, abstractmethod


class FaultModelGenerator(ABC):
    @abstractmethod
    def generate_fault_model(self, args):
        pass


class FaultModelGeneratorDiscrete(FaultModelGenerator):
    def generate_fault_model(self, args):
        mapping = eval(args)

        def fault_model(a):
            return mapping[a]

        return fault_model


class FaultModelGeneratorBox(FaultModelGenerator):
    def generate_fault_model(self, args):

        parts = args.split(";")
        arr_coef = eval(parts[0])
        arr_cnst = eval(parts[1])
        lwr_bond = eval(parts[2])
        upr_bond = eval(parts[3])

        def fault_model(a_arr):
            new_a_arr = copy.deepcopy(a_arr)
            for index in range(len(a_arr)):
                new_a_arr[index] = min(max(a_arr[index] * arr_coef[index] + arr_cnst[index], lwr_bond), upr_bond)
            return new_a_arr

        return fault_model

def same_box_action(a1, a2):
    if len(a1) != len(a2):
        return False
    for a, b in zip(a1, a2):
        if a != b:
            return False
    return True
