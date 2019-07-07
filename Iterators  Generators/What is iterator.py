# Итератор представляет собой объект перечислитель,
# который для данного объекта выдает следующий элемент, либо бросает исключение, если элементов больше нет.

# объекты, элементы которых можно перебирать в цикле for, содержат в себе объект итератор, для того,
# чтобы его получить необходимо использовать функцию iter(), а для извлечения следующего элемента из итератора – функцию next().

num_list = [1, 2, 3, 4, 5]

itr = iter(num_list)
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))

# Если нужно обойти элементы внутри объекта вашего собственного класса, необходимо построить свой итератор.
# Создадим класс, объект которого будет итератором, выдающим определенное количество единиц, которое пользователь задает
# при создании объекта. Такой класс будет содержать конструктор,
# принимающий на вход количество единиц и метод __next__(), без него экземпляры данного класса не будут итераторами.

class SimpleIterator:
    def __iter__(self):
        return self

    def __init__(self, limit):
        self.limit = limit
        self.counter = 0

    def __next__(self):
        if self.counter < self.limit:
            self.counter += 1
            return 1
        else:
            raise StopIteration

s_iter2 = SimpleIterator(5)
for i in s_iter2:
    print(i)

