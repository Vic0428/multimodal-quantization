def find_parameters_based_on_patterns(model, pattern):
    """
    Match model parameter names with pattern
    """
    model_parameters = model.state_dict()
    data = []
    for key in model_parameters.keys():
        match = pattern.match(key)
        if match:
            layer_id = match.group(1)
            data.append(model_parameters[key].view(-1).numpy())
    return data
