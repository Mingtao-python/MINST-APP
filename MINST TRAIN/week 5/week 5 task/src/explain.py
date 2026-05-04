def explain_confusions():
    print("""
Based on the confusion matrix:

- The model often confuses 8 and 9.
  Reason: both have a closed loop and similar shape.

- It also confuses 3 and 5.
  Reason: some handwritten 3s look like a curved 5.

- 1 and 7 are occasionally mixed.
  Reason: some handwritten 1s have a serif at the top.

These mistakes match the visual similarity of handwritten digits.
""")