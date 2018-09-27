import matplotlib.pyplot as plt

# Two Dice - possible results

a = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Total Number of Ways It will Occur.
# 2{(1,1)}  => 1/36
#
# 3{(1,2),(2,1)} => 2/36
#
# 4{(2,2),(3,1),(1,3)} => 3/36
#
# 5{(1,4),(4,1),(2,3),(3,2)} => 4/36
#
# 6{(3,3),(1,5),(5,1),(2,4),(4,2)} => 5/36
#
# 7{(1,6),(6,1),(2,5),(5,2),(3,4),(4,3)} => 6/36
#
# 8{(2,6),(6,2),(3,5),(5,3),(4,4)} => 5/36
#
# 9{(3,6),(6,3),(5,4),(4,5)} => 4/36
#
# 10{(4,6),(6,4),(5,5)} => 3/36
#
# 11{(5,6),(6,5)} => 2/36
#
# 12{(6,6)} = > 1/36

b = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]

s = sum(b)

print s

b = [float(x)/s for x in b]

plt.title('Binomial Distribution of Throwing Two Dices')

plt.xlabel('Result')
plt.ylabel('Probability')

plt.bar(a, b)

plt.show()