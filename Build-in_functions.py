#Returns True if all items in an iterable object are true
mylist = [True, True, True]
x = all(mylist)

#Returns True if any item in an iterable object is true
mylist = [False, True, False]
x = any(mylist)

#Returns an array of bytes
x = bytearray(4)

#Returns a bytes object
x = bytes(4)

#Returns True if the specified object is callable, otherwise False
def x():
  a = 5

print(callable(x))

#Returns a character from the specified Unicode code.
x = chr(97)

#Returns the specified source as an object, ready to be executed
x = compile('print(55)', 'test', 'eval')
exec(x)

#Returns a complex number
x = complex(3, 5)

#Deletes the specified attribute (property or method) from the specified object
class Person:
  name = "John"
  age = 36
  country = "Norway"

delattr(Person, 'age')

#Returns a dictionary (Array)
x = dict(name = "John", age = 36, country = "Norway")

#Returns a list of the specified object's properties and methods
class Person:
  name = "John"
  age = 36
  country = "Norway"

print(dir(Person))

#Returns the quotient and the remainder when argument1 is divided by argument2
x = divmod(5, 2)

#Takes a collection (e.g. a tuple) and returns it as an enumerate object
x = ('apple', 'banana', 'cherry')
y = enumerate(x)

#Evaluates and executes an expression
x = 'print(55)'
eval(x)

#Executes the specified code (or object)
x = 'name = "John"\nprint(name)'
exec(x)

#Use a filter function to exclude items in an iterable object
ages = [5, 12, 17, 18, 24, 32]

def myFunc(x):
  if x < 18:
    return False
  else:
    return True

adults = filter(myFunc, ages)

for x in adults:
  print(x)

#Formats a specified value
x = format(0.5, '%')

#Returns a frozenset object
mylist = ['apple', 'banana', 'cherry']
x = frozenset(mylist)

#Returns the value of the specified attribute (property or method)
class Person:
  name = "John"
  age = 36
  country = "Norway"

x = getattr(Person, 'age')

#Returns the current global symbol table as a dictionary
x = globals()
print(x)

#Returns True if the specified object has the specified attribute (property/method)
class Person:
  name = "John"
  age = 36
  country = "Norway"

x = hasattr(Person, 'age')

#Executes the built-in help system
help()

#Returns the id of an object
x = ('apple', 'banana', 'cherry')
y = id(x)

#Allowing user input
print('Enter your name:')
x = input()
print('Hello, ' + x)

#Returns True if a specified object is an instance of a specified object
x = isinstance(5, int)

#Returns True if a specified class is a subclass of a specified object
class myAge:
  age = 36

class myObj(myAge):
  name = "John"
  age = myAge

x = issubclass(myObj, myAge)

#Returns a list
x = list(('apple', 'banana', 'cherry'))

#Returns an updated dictionary of the current local symbol table
x = locals()
print(x)

#Returns the specified iterator with the specified function applied to each item
def myfunc(n):
  return len(n)

x = map(myfunc, ('apple', 'banana', 'cherry'))

#Returns the next item in an iterable
mylist = iter(["apple", "banana", "cherry"])
x = next(mylist)
print(x)
x = next(mylist)
print(x)
x = next(mylist)
print(x)

#Returns a new object
x = object()

#Returns a reversed iterator
alph = ["a", "b", "c", "d"]
ralph = reversed(alph)
for x in ralph:
  print(x)

#Returns a new set object
x = set(('apple', 'banana', 'cherry'))

#Returns a slice object
a = ("a", "b", "c", "d", "e", "f", "g", "h")
x = slice(2)
print(a[x])

#Returns a sorted list
a = ("b", "g", "a", "d", "f", "c", "h", "e")
x = sorted(a)
print(x)

#Sums the items of an iterator
a = (1, 2, 3, 4, 5)
x = sum(a)

#Returns a tuple
x = tuple(('apple', 'banana', 'cherry'))

#Returns an iterator, from two or more iterators
a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica")

x = zip(a, b)
