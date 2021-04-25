from typing import Any, List


def cartesian_product(fixed_parameters: dict = None, **kwargs: Any) -> List[dict]:
    fixed_parameters = {} if fixed_parameters is None else fixed_parameters
    if len(kwargs) == 0:
        return [fixed_parameters]
    else:
        cart_product = []
        parameter, values = kwargs.popitem()
        for value in values:
            new_parameters = {**fixed_parameters, parameter: value}
            sub_product = cartesian_product(fixed_parameters=new_parameters, **kwargs)
            cart_product.append(sub_product)
        return [parameters for sub_product in cart_product for parameters in sub_product]


def incremental_levels(num_levels, parameters):
    outputs = [{}]
    for levels in range(num_levels):
        outputs += cartesian_product(**{str(level): parameters for level in range(levels + 1)})
    outputs = [tuple(h.values()) for h in outputs]
    return outputs
