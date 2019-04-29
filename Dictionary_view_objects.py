# The objects returned by dict.keys(), dict.values() and dict.items() are view objects.
# They provide a dynamic view on the dictionary’s entries, which means that when the dictionary changes, the view reflects these changes.

dishes = {'eggs': 2, 'sausage': 1, 'bacon': 1, 'spam': 500}
keys = dishes.keys()
values = dishes.values()

# view objects are dynamic and reflect dict changes
del dishes['eggs']
keys  # No eggs anymore!
dict_keys(['sausage', 'bacon', 'spam'])

values  # No eggs value (2) anymore!
dict_values([1, 1, 500])