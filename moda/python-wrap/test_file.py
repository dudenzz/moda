import moda
import numpy as np
ds1 = moda.DataSet()
p1 = moda.Point([1,2,3,4])
p2 = moda.Point([3,4,5,6])

ds1.add(p1)
ds1.add(p2)

print(ds1)
data = np.array([[1,2,3],[3,4,5]])
data = data.astype(np.float64)
ds2 = moda.DataSet(data)

print(ds2)

ds3 = moda.DataSet('linear_d4n100_1')

print(ds3)