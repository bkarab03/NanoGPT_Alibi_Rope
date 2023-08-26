from data.squad.prepare import tokenizer


def print_human_readable(X, Y, num_items=1):
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    for i in range(num_items):
        print("Original sequence:")
        print(tokenizer.decode(Y[i], skip_special_tokens=False))
        print()
        print("-----------------")

        print("Input sequence (masked):")
        print(tokenizer.decode(X[i], skip_special_tokens=False))
        print("-----------------")
