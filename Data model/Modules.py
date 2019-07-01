# A module allows you to logically organize your Python code.
# Grouping related code into a module makes the code easier to understand and use.
# A module is a Python object with arbitrarily named attributes that you can bind and reference.
# Simply, a module is a file consisting of Python code.
# A module can define functions, classes and variables. A module can also include runnable code.

# import module1[, module2[,... moduleN]

# When you import a module, the Python interpreter searches for the module in the following sequences −
# The current directory.
# If the module isn't found, Python then searches each directory in the shell variable PYTHONPATH.
# If all else fails, Python checks the default path. On UNIX, this default path is normally /usr/local/lib/python/.
# The module search path is stored in the system module sys as the sys.path variable.
#  The sys.path variable contains the current directory, PYTHONPATH, and the installation-dependent default.

# A package is a hierarchical file directory structure that defines a single Python application environment that consists of modules and subpackages and sub-subpackages, and so on.
# Consider a file Pots.py available in Phone directory. This file has following line of source code

def Pots():
   print ("I'm Pots Phone")

# Similar way, we have another two files having different functions with the same name as above −
#
# Phone/Isdn.py file having function Isdn()
#
# Phone/G3.py file having function G3()
#
# Now, create one more file __init__.py in Phone directory −
#
# Phone/__init__.py
# To make all of your functions available when you've imported Phone,
# you need to put explicit import statements in __init__.py as follows

from Pots import Pots
from Isdn import Isdn
from G3 import G3

# After you add these lines to __init__.py, you have all of these classes available when you import the Phone package.

#!/usr/bin/python

# Now import your Phone Package.
import Phone

Phone.Pots()
Phone.Isdn()
Phone.G3()

# When the above code is executed, it produces the following result −
# I'm Pots Phone
# I'm 3G Phone
# I'm ISDN Phone

# In the above example, we have taken example of a single functions in each file,
# but you can keep multiple functions in your files. You can also define different
# Python classes in those files and then you can create your packages out of those classes.