def check_types(obj, types):
    if isinstance(types, type):
        types = (types,)
    if not isinstance(obj, types):
        raise TypeError(f"Expected type: {types}, got: {type(obj)}")
