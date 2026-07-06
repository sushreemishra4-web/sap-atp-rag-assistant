def even_num():
    n = 2

    while n >= 2:
        yield n
        n = n + 2

counter = 0
for item in even_num():
    print(item)

    counter += 1
    if counter == 5:
        break
