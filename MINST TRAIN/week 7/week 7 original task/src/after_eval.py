def a_eval(user_input):
    try:
        weird_chars = [
            "@", "#", "$", "%", "^", "&", "_", "=", "~", "`",
            "{", "}", "[", "]", "<", ">", ":", ";", "?", "!", "|", "\\", "'", "\"", ",", ".",
            "×", "÷", "±", "√", "∞", "∑", "∏", "∫", "≈", "≠", "≤", "≥",
            "＋", "－", "＊", "／", "（", "）", "＝",
            "\u200b", "\u3000", "\u200c", "\u200d", "\t", "\n", "\r", " ",
        ]

        weird_chars += [chr(i) for i in range(ord('a'), ord('z')+1)]
        weird_chars += [chr(i) for i in range(ord('A'), ord('Z')+1)]

        bad_chars = []

        for ch in weird_chars:
            if ch in user_input:
                bad_chars.append(ch)

        if bad_chars:
            raise ValueError(f"Invalid characters detected: {', '.join(bad_chars)}")

        if "**" in user_input:
            raise ValueError("Exponent not allowed")

        return eval(user_input)

    except Exception as e:
        print(f"An error had occured, error details: {e}")
        return None
