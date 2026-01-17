import one as one

# invalid script: name 2_one
# invalid script: name 2one
#    import 2one as one
#           ^
#SyntaxError: invalid decimal literal

# invalid scrip name: one.two.py
#    import one.two as one
# ModuleNotFoundError: No module named 'one.two'; 'one' is not a package

# invalid scrip name: one-two.py
#    import one-two as one
#              ^
# SyntaxError: invalid syntax

# invalid script name: one script.py
#    import one script as one
#               ^^^^^^
# SyntaxError: invalid syntax

# invalid script name: one,script;xxx.py
#    import one,script;xxx as one
#                          ^^
# SyntaxError: invalid syntax

# valid script names
# _one.py
# script_1.py
# one11.py
# _1.py
# one.py

one.printPi()