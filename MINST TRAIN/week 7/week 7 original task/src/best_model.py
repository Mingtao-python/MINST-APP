def choose_best_model(results):
    best_name = None
    best_model = None
    best_acc = 0

    for name, (model, acc) in results.items():
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    print(f"\n Best model: {best_name} (accuracy = {best_acc:.4f})")
    return best_model