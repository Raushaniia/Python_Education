# Fibonacci numbers module

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()

def fib2(n):   # return Fibonacci series up to n
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result

# Now enter the Python interpreter and import this module with the following command:
# import fibo

# Packages are a way of structuring Python’s module namespace by using “dotted module names”. For example,
# the module name A.B designates a submodule named B in a package named A. Just like the use of modules saves the authors of
# different modules from having to worry about each other’s global variable names, the use of dotted module names saves the authors
# of multi-module packages like NumPy or
# Pillow from having to worry about each other’s module names.