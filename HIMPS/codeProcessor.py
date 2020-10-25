import cProfile
import pstats
import io
from pstats import SortKey

file = input('Masukkan Nama File: ')
f = open(file, 'r')

pr = cProfile.Profile(builtins=False)
pr.enable()
exec(f.read())
pr.disable()
s = io.StringIO()
with open("output.csv", "w") as f:
    ps = pstats.Stats(pr, stream=f).sort_stats('calls').strip_dirs()
    ps.print_stats()
# print(s.getvalue())
# x = f.read()
# y = x.splitlines()

# z = []
# for i in range(len(y)):
#     if (y[i] != ""):
#         z.append(y[i])


# print(z)
# To print the content of the whole file cv
# for i in range(len(z)):
#    print(z[i])

# To read only one line
# print(f.readline())
