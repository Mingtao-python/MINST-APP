#tasks:
#show 5 random images to make sure the data is loaded correctly - finished -
#make a list of the number of each numbers and draw a bar chart accurately - finished -
#show the average image of each number (0-9) and make sure they look like the correct number - finished -
#declare the number that the MINST model could mistake the most and explain why - finished -

from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
for i in range(10):
    digit = i
    
    images = digits.images[y == digit]
    mean_image = np.mean(images, axis=0)

    plt.imshow(mean_image, cmap='gray')
    plt.title(f'Average Digit Of {str(digit)} (Average picture)')
    plt.show()

for i in range(5):
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Example Digit {str(i)}")
    plt.show()

counts = pd.Series(y).value_counts().sort_index()
plt.bar(counts.index, counts.values)
plt.title("Number counts")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()
#-----------Explaining the mosts mistaken number----------------
print('''
There are many numbers that the MINST model can miss. They are mostly:
- 1 and 7: as they look similar, especially some handwritten 1's contains a small horizontal line at the top, witch can be easily mistaken for a 7.
- 3 and 8: as they also look similar, especially some handwritten 3's can be written with a closed loop at the top, making them look like 8's.
- 4 and 9: as some handwritten 4's has a circular shape at the top, making them look like 9's.
- 5 and 6: as some handwritten 5's can be written with a closed circle at the bottom, making them look like 6's.
''')

print('Thank you for reading.')