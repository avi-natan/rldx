from h_fault_model_generator import FaultModelGeneratorDiscrete, FaultModelGeneratorBox

fault_model_generators = {
    # abstract fault models (suitable for all environments)
    "discrete": FaultModelGeneratorDiscrete(),
    "box": FaultModelGeneratorBox(),
}
