# Декораторы позволяют нам обернуть другую функцию, чтобы расширить поведение обернутой функции, не изменяя ее постоянно.
#
# В Decorators функции берутся в качестве аргумента другой функции и затем вызываются внутри функции-оболочки.

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

def say_whee():
    print("Whee!")

say_whee = my_decorator(say_whee)

