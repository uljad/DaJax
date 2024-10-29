#print jax tree
def print_shapes(tree, prefix=''):
    if isinstance(tree, dict):
        for key, value in tree.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            print_shapes(value, new_prefix)
    else:
        print(f"{prefix}: {tree}")