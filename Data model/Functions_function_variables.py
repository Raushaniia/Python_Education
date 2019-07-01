# Все параметры (аргументы) в языке Python передаются по ссылке.
# Это означает, что если вы изменяете то, на что ссылается параметр в функции, это изменение также отражается в вызывающей функции.

# Function definition is here
def changeme( mylist ):
   "This changes a passed list into this function"
   mylist = [1,2,3,4]; # This would assig new reference in mylist
   print ("Values inside the function: ", mylist)
   return

# Now you can call changeme function
mylist = [10,20,30];
changeme( mylist );
print ("Values outside the function: ", mylist)


# Анонимные функции  не объявляются стандартным способом с помощью ключевого слова def. Вы можете использовать ключевое слово lambda для создания небольших анонимных функций.
#
# Лямбда-формы могут принимать любое количество аргументов, но возвращать только одно значение в форме выражения. Они не могут содержать команды или несколько выражений.
#
# Анонимная функция не может быть прямым вызовом для печати, потому что лямбда-выражение требует выражения
#
# Лямбда-функции имеют свое собственное локальное пространство имен и не могут получить доступ к переменным, отличным от переменных в их списке параметров и переменных в глобальном пространстве имен.
#
# Хотя кажется, что лямбда-выражения - это однострочная версия функции, они не эквивалентны встроенным операторам в C или C ++, целью которых является передача выделения стека функций во время вызова по соображениям производительности.

# The syntax of lambda functions contains only a single statement, which is as follows −
# lambda [arg1 [,arg2,.....argn]]:expression

# Function definition is here
sum = lambda arg1, arg2: arg1 + arg2;

# Now you can call sum as a function
print ("Value of total : ", sum( 10, 20 ))
print ("Value of total : ", sum( 20, 20 ))