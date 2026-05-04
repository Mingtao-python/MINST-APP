def a_int(user_input):
    if user_input.isdigit():
        if 0 <= int(user_input) < 10:
            if float(user_input) % 1 == 0:
                storage = int(user_input)
                print('Saved correctly')
            else:
                print('Input not valid, not saved(it have to be an integer)')
        else:
            print('Input not valid, not saved(it have to be a number between 0 and 9)')
    else:
        print('Input not valid, not saved(it have to be a number)')