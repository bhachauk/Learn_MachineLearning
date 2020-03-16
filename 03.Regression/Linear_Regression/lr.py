import numpy as np
import warnings
import random

from sklearn import linear_model
warnings.filterwarnings("ignore")



# Case -1
# a = range(20)
# b = [x+1 for x in a]

# Case - 2
# a = range(20)
# a = [random.randrange(50) for _ in a]
# b = [x+1 for x in a]


# Case - 3
a = range(50)
a = [random.randrange(100) for _ in a]
b = [x+1 if x % 2 == 0 else x-1 for x in a]

input_data = np.array(a).reshape(-1, 1)
target_data = np.array(b).reshape((-1, 1))

print("Input:")
for i, x in enumerate(input_data):
    print(input_data[i], target_data[i])

lr = linear_model.LinearRegression()

lr.fit(input_data, target_data)
val = lr.predict([[1004]])
print ('predicted : ', val)

# 1 - Recap (Overview)
# 2 - Hands_on_models (Models related intro - Example : sk_learn)
# 3 - Numbering system learning. (Linear, random)
# 4 - Non-linearity models (Milk - problem, fitting category, accuracy : rmse )
