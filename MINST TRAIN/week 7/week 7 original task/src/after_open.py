def a_open(user_input):
    try:
        with open(user_input) as _:
            print('File opened successfully')
    except Exception as e:
        print(f'An error had occured, error details: {e}')