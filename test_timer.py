from hwcounter import count, count_end
import time

st = count()
time.sleep(10)
et = count_end()

print((et - st)/1e9)
