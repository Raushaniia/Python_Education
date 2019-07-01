# Introduction
# id() is an inbuilt function in Python.
# Syntax:
#
# id(object)
#
# As we can see the function accepts a single parameter and is used to return the identity of an object.
# This identity has to be unique and constant for this object during the lifetime. Two objects with non-overlapping lifetimes may have the same id() value.
# If we relate this to C, then they are actually the memory address, here in Python it is the unique id. This function is generally used internally in Python.


# This program shows various identities
str1 = "geek"
print(id(str1))

str2 = "geek"
print(id(str2))

# This will return True
print(id(str1) == id(str2))

# Use in Lists
list1 = ["aakash", "priya", "abdul"]
print(id(list1[0]))
print(id(list1[2]))

# This returns false
print(id(list1[0]) == id(list1[2]))
