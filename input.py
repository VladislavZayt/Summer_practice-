import random

# генерация n случайных точек. 
def generate_random_points(n):
    points = []
    for i in range(n):
        point = (random.uniform(0, 10), random.uniform(0, 10))
        points.append(point)
    return points

# загрузка n точек из файла
def load_data(filename, n):
    points = []
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            n = min(n, len(lines))
            for line in lines[:n]:
                if not line:
                    break
                point = tuple(map(float, line.split()))
                points.append(point)
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Точки сгенерируются случайно.")
        points = generate_random_points(n)
    return points