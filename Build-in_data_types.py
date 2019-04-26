# Built-in Data types[edit]
# Python's built-in (or standard) data types can be grouped into several classes. Sticking to the hierarchy scheme used in the official Python documentation these are numeric types, sequences, sets and mappings (and a few more not discussed further here). Some of the types are only available in certain versions of the language as noted below.
# boolean: the type of the built-in values True and False. Useful in conditional expressions, and anywhere else you want to represent the truth or falsity of some condition. Mostly interchangeable with the integers 1 and 0. In fact, conditional expressions will accept values of any type, treating special ones like boolean False, integer 0 and the empty string "" as equivalent to False, and all other values as equivalent to True. But for safety’s sake, it is best to only use boolean values in these places.
# Numeric types:
# int: Integers; equivalent to C longs in Python 2.x, non-limited length in Python 3.x
# long: Long integers of non-limited length; exists only in Python 2.x
# float: Floating-Point numbers, equivalent to C doubles
# complex: Complex Numbers
# Sequences:
# str: String; represented as a sequence of 8-bit characters in Python 2.x, but as a sequence of Unicode characters (in the range of U+0000 - U+10FFFF) in Python 3.x
# bytes: a sequence of integers in the range of 0-255; only available in Python 3.x
# byte array: like bytes, but mutable (see below); only available in Python 3.x
# list
# tuple
# Sets:
# set: an unordered collection of unique objects; available as a standard type since Python 2.6
# frozen set: like set, but immutable (see below); available as a standard type since Python 2.6
# Mappings:
# dict: Python dictionaries, also called hashmaps or associative arrays, which means that an element of the list is associated with a definition, rather like a Map in Java
# Some others, such as type and callables

# http://www.informit.com/articles/article.aspx?p=453682&seqNum=5