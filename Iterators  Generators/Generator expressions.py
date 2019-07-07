# In Python, generators provide a convenient way to implement the iterator protocol. Generator is an iterable created using a
# function with a yield statement.
#
# The main feature of generator is evaluating the elements on demand. When you call a normal function with a return statement
# the function is terminated whenever it encounters a return statement. In a function with a yield statement
# the state of the function is “saved” from the last call and can be picked up the next time you call a generator function.

# In terms of syntax, the only difference is that you use parentheses instead of square brackets.
# However, the type of data returned by list comprehensions and generator expressions differs.

# There are various other expressions that can be simply coded similar to list comprehensions but instead of brackets we
# # use parenthesis. These expressions are designed for situations where the generator is used right away by an enclosing function.
# # Generator expression allows creating a generator without a yield keyword.
# # However, it doesn’t share the whole power of generator created with a yield function. Example :

# Python code to illustrate generator expression
generator = (num ** 2 for num in range(10))
for num in generator:
	print(num)

# We can also generate a list using generator expressions :
string = 'geek'
li = list(string[i] for i in range(len(string)-1, -1, -1))
print(li)

