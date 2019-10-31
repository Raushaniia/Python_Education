# Генератор – это функция, которая будучи вызванной в функции next() возвращает следующий объект согласно алгоритму ее работы.
# Вместо ключевого слова return в генераторе используется yield.
# Проще всего работу генератор посмотреть на примере. Напишем функцию, которая генерирует необходимое нам количество единиц.

def simple_generator(val):
   while val > 0:
       val -= 1
       yield 1

gen_iter = simple_generator(5)
print(next(gen_iter))
print(next(gen_iter))
print(next(gen_iter))
print(next(gen_iter))
print(next(gen_iter))
print(next(gen_iter))

# Ключевым моментом для понимания работы генераторов является то, при вызове yield функция не прекращает свою работу,
# а “замораживается” до очередной итерации, запускаемой функцией next(). Если вы в своем генераторе, где-то используете
# ключевое слово return, то дойдя до этого места будет выброшено исключение StopIteration,
# а если после ключевого слова return поместить какую-либо информацию, то она будет добавлена к описанию StopIteration.