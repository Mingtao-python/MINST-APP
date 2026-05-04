def explanation(wrong_idx, y_test):
    best_score = 0
    best_key = None
    explain = {1: "- 1 and 7 are occasionally mixed.\n  Reason: some handwritten 1s have a serif at the top.",
               2: "- 2 and 8 are sometimes confused.\n  Reason: some handwritten 2s have a loop that can look like an 8.",
               3: "- sometimes 3 and 5 are confused.\n  Reason: some handwritten 3s look like a curved 5.",
               4: "- 4 and 9 are occasionally mixed.\n  Reason: some handwritten 4s have a closed top that can resemble a 9.",
               5: "- sometimes 3 and 5 are confused.\n  Reason: some handwritten 3s look like a curved 5.",
               6: "- 0 and 6 are sometimes confused.\n  Reason: some handwritten 0s have a small gap that can look like a 6.",
               7: "- 1 and 7 are occasionally mixed.\n  Reason: some handwritten 1s have a serif at the top.",
               8: "- The model often confuses 8 and 9.\n  Reason: both have a closed loop and similar shape.",
               9: "- The model often confuses 8 and 9.\n  Reason: both have a closed loop and similar shape."}
    scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for i, idx in enumerate(wrong_idx[:8]):
        scores[y_test[idx]] += 1
    for key in scores:
        if scores[key] >= 2:
            print(explain[key])
    for key in scores:
        if scores[key] > best_score:
            best_score = scores[key]
            best_key = str(key)
        elif scores[key] == best_score and best_key is not None:
            best_key += f" and {key}"
    if best_key is not None:
        print(f"The most frequently misclassified class is {best_key} with {best_score} misclassifications.")