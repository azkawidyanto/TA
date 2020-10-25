import io
import pstats
import cProfile
from pstats import SortKey


# code you want to profile


# pr = cProfile.Profile()
# pr.enable()
# ... do something ...
x = 5
y = 10

x += y
y = x-y
x -= y
print('The value of x after swapping: {}'.format(x))
print('The value of y after swapping: {}'.format(y))
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
