# An object is hashable if it has a hash value which never changes during its lifetime (it needs a __hash__() method), and can be compared to other objects (it needs an __eq__() or __cmp__() method). Hashable objects which compare equal must have the same hash value.
#
# Hashability makes an object usable as a dictionary key and a set member, because these data structures use the hash value internally.
#
# All of Python’s immutable built-in objects are hashable, while no mutable containers (such as lists or dictionaries) are. Objects which are instances of user-defined classes are hashable by default; they all compare unequal, and their hash value is their id().

tuple_a = (1,2,3)
tuple_a.__hash__()
#2528502973977326415
tuple_b = (2,3,4)
tuple_b.__hash__()
#3789705017596477050
tuple_c = (1,2,3)
tuple_c.__hash__()
#2528502973977326415
id(a) == id(c)  # a and c same object?
#False
a.__hash__() == c.__hash__()  # a and c same value?
#True

#Two instances of the same class will have two different hash values. For example:

class MyClass:
    def __init__(self, value):
        self.value = value

my_obj = MyClass(1)
print(my_obj.__hash__()) # 8757243744113
my_new_obj = MyClass(1)
print(my_new_obj.__hash__()) # -9223363279611078919

#Dictionaries in Python are using for defining their keys. They do not only look at the hash value, they also look whether the keys are the same or not. If they are not, they will be assigned to a new element instead of the same one.