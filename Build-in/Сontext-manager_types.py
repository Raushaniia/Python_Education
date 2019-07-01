# Context managers allow you to allocate and release resources precisely when you want to.
# The most widely used example of context managers is the with statement.
# Suppose you have two related operations which you’d like to execute as a pair, with a block of code in between.
# Context managers allow you to do specifically that. For example:

with open('some_file', 'w') as opened_file:
    opened_file.write('Hola!')

# The above code opens the file, writes some data to it and then closes it.
# If an error occurs while writing the data to the file, it tries to close it. The above code is equivalent to:

file = open('some_file', 'w')
try:
    file.write('Hola!')
finally:
    file.close()

#Custom file-opening Context Manager:

class File(object):
    def __init__(self, file_name, method):
        self.file_obj = open(file_name, method)
    def __enter__(self): #rquared
        return self.file_obj
    def __exit__(self, type, value, traceback):#rquared
        self.file_obj.close()