import random

# генерация n случайных точек. 
def generate_random_points(n):
    points = []
    for i in range(n):
        point = (random.uniform(0, 10), random.uniform(0, 10))
        points.append(point)
    return points

# загрузка M окружностей и n точек из файла
def load_data(filename, n):
    points = []
    M = 0 # число окружностей
    try:
        with open(filename, 'r') as file:
            M = int(file.readline())
            for i in range(n):
                line = file.readline()
                if not line:
                    break
                point = tuple(map(float, line.split()))
                points.append(point)
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Точки сгенерируются случайно.")
        print("Введите число окружностей: ", end='')
        M = int(input())
        points = generate_random_points(n)
    return M, points