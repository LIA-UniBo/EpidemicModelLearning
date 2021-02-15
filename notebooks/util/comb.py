def cartesian_product(multiple_parameters: list, parameters: dict):
    if len(multiple_parameters) == 0:
        return [parameters]
    else:
        cart_product = []
        parameter, values = multiple_parameters[0]
        for value in values:
            new_parameters = {**parameters, parameter: value}
            sub_product = cartesian_product(multiple_parameters[1:], new_parameters)
            cart_product.append(sub_product)
        return [parameters for sub_product in cart_product for parameters in sub_product]
