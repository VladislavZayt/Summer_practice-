import random

class GeneticAlgorithm:
    def __init__(self, points, params):
        self.points = points  # список (x, y)
        self.params = params  # dict с параметрами GA
        self.M = self.params.get('circles_count')  # число окружностей
        self.bounding_box = self._compute_bounding_box(points)
        self.population = []  # список особей: каждая особь — list длины 3*M
        self.history = []  # для сохранения поколений (если нужно)
        self.current_generation = 0
        self.attempts_limit = 1000
        self.pop_size = 50
        self.r_min = 1.0
        self.r_max = 30.0
        self.crossover_tries = 5
        self.stats = {'best': [], 'average': []}
        self.current_fitness = []
        self.crossover_type = self.params.get('crossover_type')
        self.mutation_type = self.params.get('mutation_type')

    def _compute_bounding_box(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # Возвращаем (xmin, xmax, ymin, ymax)
        return (min(xs), max(xs), min(ys), max(ys))

    def _has_intersections(self, individual):
        # Проверяет, есть ли пересечения между любыми двумя окружностями в individual
        for i in range(self.M):
            xi, yi, ri = individual[3 * i], individual[3 * i + 1], individual[3 * i + 2]
            for j in range(i + 1, self.M):
                xj, yj, rj = individual[3 * j], individual[3 * j + 1], individual[3 * j + 2]
                dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
                # Если центр ближе, чем сумма радиусов → пересечение
                if dist2 < (ri + rj) ** 2:
                    return True
        return False

    def initialize_population(self):
        """
        Генерирует популяцию из pop_size особей, каждая с M непересекающимися окружностями.
        Если не удаётся за attempts_limit попыток для очередной особи, выдаём предупреждение.
        """
        self.population = []
        attempts_limit = self.attempts_limit
        xmin, xmax, ymin, ymax = self.bounding_box
        pop_size = self.pop_size

        for idx in range(pop_size):
            created = False
            for attempt in range(attempts_limit):
                individual = []
                circles = []
                valid = True
                # Генерируем по одной окружности, проверяя пересечения
                for _ in range(self.M):
                    x = random.uniform(xmin, xmax)
                    y = random.uniform(ymin, ymax)
                    # Радиус: можно задать минимум, например 1.0, или 0. Если 0, окружность бесполезна.
                    r = random.uniform(self.r_min, self.r_max)
                    # Проверяем пересечение с уже сгенерированными:
                    for (cx, cy, cr) in circles:
                        if (x - cx) ** 2 + (y - cy) ** 2 < (r + cr) ** 2:
                            valid = False
                            break
                    if not valid:
                        break
                    circles.append((x, y, r))
                if valid and len(circles) == self.M:
                    # Собираем individual
                    for (x, y, r) in circles:
                        individual.extend([x, y, r])
                    self.population.append(individual)
                    created = True
                    break
            if not created:
                print(f"Warning: не удалось сгенерировать валидную особь #{idx+1} за {attempts_limit} попыток.")
                # Можно: либо снизить требование и добавить "пустую" или предыдущую особь,
                # либо просто пропустить. Здесь пропустим, но размер популяции станет меньше.
        # Если population меньше pop_size, можно сгенерировать дубликаты лучших,
        # чтобы довести до размера:
        if len(self.population) < pop_size and self.population:
            needed = pop_size - len(self.population)
            best = max(self.population, key=self._fitness_no_penalty)
            for _ in range(needed):
                self.population.append(best.copy())
            print(f"Популяция увеличена дубликатами лучшей особи, чтобы достичь pop_size={pop_size}.")

        self.current_generation = 0
        # Храним копию начальной популяции, если нужно для шага назад
        self.history = [list(self.population)]
        self.stats = {'best': [], 'average': []}
        # Первая оценка
        self.current_fitness = self.evaluate_population()


    def evaluate_population(self):
        fitness_values = []
        for ind in self.population:
            fit = self._fitness_no_penalty(ind)
            fitness_values.append(fit)
        best = max(fitness_values)
        avg = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
        self.stats['best'].append(best)
        self.stats['average'].append(avg)
        self.current_fitness = fitness_values
        return fitness_values

    def _fitness_no_penalty(self, individual):
        # Подсчитываем число покрытых точек; пересечений не должно быть
        cover_count = 0
        for (px, py) in self.points:
            for i in range(self.M):
                x, y, r = individual[3 * i], individual[3 * i + 1], individual[3 * i + 2]
                if (px - x) ** 2 + (py - y) ** 2 <= r ** 2:
                    cover_count += 1
                    break
        return cover_count

    def select(self):
        # Турнирная селекция без ошибок: предполагается, что current_fitness уже установлен
        k = self.params.get('tournament_size', 3)
        pop_len = len(self.population)
        if pop_len == 0:
            raise ValueError("Популяция пуста при селекции!")
        # Если k > pop_len, уменьшаем k
        if k > pop_len:
            k = pop_len
        indices = random.sample(range(pop_len), k)
        best_idx = max(indices, key=lambda i: self.current_fitness[i])
        return self.population[best_idx].copy()
    
    def crossover(self, p1, p2, type_flag):
        """
        Функция скрещевания
        type_flag == 0 => Одноточечное скрещивание
        type_flag == 1 => Двухточечное скрещивание
        type_flag == 2 => Равномерное скрещивание
        type_flag == 3 => Скрещивание смешением
        """
        def make_children_one_point(pt1, pt2, point):
            child1 = pt1[:point] + pt2[point:]
            child2 = pt2[:point] + pt1[point:]
            return child1, child2
    
        def make_children_two_points(pt1, pt2, point, point2):           
            child1 = pt1[:point] + pt2[point:point2] + pt1[point2:]
            child2 = pt2[:point] + pt1[point:point2] + pt2[point2:]
            return child1, child2

        def make_children_uniform(pt1, pt2):
            child1 = []
            child2 = [] 
            for i in range(0, len(pt1)):
                select_chance = random.randint(0, 1)
                if select_chance == 1:
                    child1.append(pt2[i])
                    child2.append(pt1[i])
                else:
                    child1.append(pt1[i])
                    child2.append(pt2[i])
            return child1, child2

        def make_childeren_bias(pt1, pt2, alpha=0.5):
            child1 = []
            child2 = []
            for i in range(0, len(pt1)):
                min_val = min(pt1[i], pt2[i])
                max_val = max(pt1[i], pt2[i])

                lower_bound = min_val - alpha * (max_val - min_val)
                upper_bound = max_val + alpha * (max_val - min_val)

                child1.append(random.uniform(lower_bound, upper_bound))
                child2.append(random.uniform(lower_bound, upper_bound))

            return child1, child2

        max_tries = self.crossover_tries
        # Функция для попытки кроссовера один раз:
        for _ in range(max_tries):
                # Выбираем точку скрещивания: между окружностями
            if type_flag == 0:
                cut_idx = random.randint(1, self.M - 1) * 3
                child1, child2 = make_children_one_point(p1, p2, cut_idx)
                # Проверяем пересечения:
            elif type_flag == 1:
                cut_idx_1 = random.randint(1, self.M - 1) * 3
                cut_idx_2 = random.randint(1, self.M - 1) * 3
                while cut_idx_1 == cut_idx_2:
                    cut_idx_1 = random.randint(1, self.M - 1) * 3
                    cut_idx_2 = random.randint(1, self.M - 1) * 3
                if cut_idx_1 > cut_idx_2:
                    cut_idx_1, cut_idx_2 = cut_idx_2, cut_idx_1

                child1, child2 = make_children_two_points(p1, p2, cut_idx_1, cut_idx_2)
            elif type_flag == 2:
                child1, child2 = make_children_uniform(p1, p2)
            elif type_flag == 3:
                child1, child2 = make_childeren_bias(p1, p2)
            else:
                print("Error!")
            ok1 = not self._has_intersections(child1)
            ok2 = not self._has_intersections(child2)
            if ok1 and ok2:
                return child1, child2
                # Если один из детей валиден, другой нет, можно делать: возвращать одного ребёнка + копию родителя
            if ok1 and not ok2:
                    # child1 валиден, child2 нет
                return child1, p1.copy()
            if ok2 and not ok1:
                return child2, p2.copy()
                # иначе оба невалидны — пробуем другую точку скрещивания
            # Если не удалось за max_tries, возвращаем родителей (без изменений)
        return p1.copy(), p2.copy()

    def mutate(self, individual, type_flag):
        """
        Функция мутированния
        type_flag == 0 => Вещественная мутация
        type_flag == 1 => Мутация радиуса
        type_flag == 2 => Мутация слиянием 
        type_flag == 3 => Мутация равномерным шумом 
        type_flag == 4 => Комбинированная мутация (Гауссовский шум для координат центра, равномерный для радиуса)
        Если после мутирования окружности пересекаются, возвращается исходное состояние.
        """
        original = individual.copy()

        def real_mutation(individual):
            xmin, xmax, ymin, ymax = self.bounding_box
            changed = False
            for i in range(self.M):
                if random.random() < self.params.get('mutation_rate', 0.1):
                    # Добавляем шум; затем нужно скорректировать, чтобы центр оставался в bounding_box,
                    # радиус оставался в [r_min, r_max]
                    xi = individual[3 * i] + random.gauss(0, self.params.get('sigma', 1.0))
                    yi = individual[3 * i + 1] + random.gauss(0, self.params.get('sigma', 1.0))
                    ri = individual[3 * i + 2] + random.gauss(0, self.params.get('sigma', 1.0))
                    # Clamp:
                    xi = min(max(xi, xmin), xmax)
                    yi = min(max(yi, ymin), ymax)
                    ri = min(max(ri, self.r_min), self.r_max)
                    individual[3 * i] = xi
                    individual[3 * i + 1] = yi
                    individual[3 * i + 2] = ri
                    changed = True
            return individual, changed

        def radius_mutation(individual):
            changed = False
            for i in range(self.M):
                if random.random() < self.params.get('mutation_rate', 0.1):
                    ri = individual[3 * i + 2] + random.gauss(0, self.params.get('sigma', 1.0))
                    ri = min(max(ri, self.r_min), self.r_max)
                    individual[3 * i + 2] = ri
                    changed = True
            return individual, changed

        def distance(circle_1, circle_2):
            x1, y1, r1 = circle_1
            x2, y2, r2 = circle_2
    
            dx = x2 - x1
            dy = y2 - y1
            centers_distance = (dx**2 + dy**2)**0.5
    
            circle_distance = centers_distance - r1 - r2
            return max(0, circle_distance)
        def merge_mutation(individual):
            changed = False
            if random.random() < self.params.get('mutation_rate', 0.1):
                best_pair = (0, 1)
                min_dist = 100000000000000000
                for i in range(self.M):
                    for j in range(i+1, self.M):
                        dist = distance(individual[3*i:3*i+3], individual[3*j:3*j+3])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (i, j)
    
                i, j = best_pair
                # Заменяем одну окружность на среднюю, другую - на случайную
                xmin, xmax, ymin, ymax = self.bounding_box
                individual[3*i:3*i+3] = [(individual[3*i]+individual[3*j])/2, (individual[3*i+1]+individual[3*j+1])/2, max(individual[3*i+2], individual[3*j+2])*1.1]
                individual[3*j:3*j+3] = [random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(self.r_min, self.r_max)]
                changed = True
            return individual, changed
        
        def uniform_mutation(individual):
            xmin, xmax, ymin, ymax = self.bounding_box
            changed = False
            for i in range(self.M):
                if random.random() < self.params.get('mutation_rate', 0.1):
                    xi = individual[3 * i] + random.uniform(-self.params.get('delta', 2.0), self.params.get('delta', 2.0))
                    yi = individual[3 * i + 1] + random.uniform(-self.params.get('delta', 2.0), self.params.get('delta', 2.0))
                    ri = individual[3 * i + 2] + random.uniform(-self.params.get('delta', 2.0), self.params.get('delta', 2.0))
                    # Clamp:
                    xi = min(max(xi, xmin), xmax)
                    yi = min(max(yi, ymin), ymax)
                    ri = min(max(ri, self.r_min), self.r_max)
                    individual[3 * i] = xi
                    individual[3 * i + 1] = yi
                    individual[3 * i + 2] = ri
                    changed = True
            return individual, changed
        def combine_mutation(individual):
            xmin, xmax, ymin, ymax = self.bounding_box
            changed = False
            for i in range(self.M):
                if random.random() < self.params.get('mutation_rate', 0.1):
                    xi = individual[3 * i] + random.gauss(0, self.params.get('sigma', 1.0))
                    yi = individual[3 * i + 1] + random.gauss(0, self.params.get('sigma', 1.0))
                    ri = individual[3 * i + 2] + random.uniform(-self.params.get('delta', 2.0), self.params.get('delta', 2.0))
                    # Clamp:
                    xi = min(max(xi, xmin), xmax)
                    yi = min(max(yi, ymin), ymax)
                    ri = min(max(ri, self.r_min), self.r_max)
                    individual[3 * i] = xi
                    individual[3 * i + 1] = yi
                    individual[3 * i + 2] = ri
                    changed = True
            return individual, changed


        if type_flag == 0:
            individual, changed = real_mutation(individual)

        if type_flag == 1:
            individual, changed = radius_mutation(individual)

        if type_flag == 2:
            individual, changed = merge_mutation(individual)
        if type_flag == 3:
            individual, changed = uniform_mutation(individual)
        if type_flag == 4:
            individual, changed = combine_mutation(individual)
        if changed:
            # Проверка пересечений:
            if self._has_intersections(individual):
                # Откатываем
                individual[:] = original[:]

    def step(self):
        # Оценка текущего поколения
        self.current_fitness = self.evaluate_population()

        new_population = []
        pop_size = self.pop_size

        # Элитарность: сохраняем лучшую особь
        if self.population:
            best_idx = self.current_fitness.index(max(self.current_fitness))
            new_population.append(self.population[best_idx].copy())

        # Генерация остальных особей
        while len(new_population) < pop_size:
            parent1 = self.select()
            parent2 = self.select()
            # Кроссовер
            if random.random() < self.params.get('crossover_rate', 0.7):
                child1, child2 = self.crossover(parent1, parent2, self.crossover_type)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Мутация
            self.mutate(child1, self.mutation_type)
            if len(new_population) < pop_size:
                new_population.append(child1)
            if len(new_population) < pop_size:
                self.mutate(child2, self.mutation_type)
                new_population.append(child2)

        # Обновляем состояние
        self.population = new_population
        self.current_generation += 1
        self.history.append(list(self.population))

        # Оценка нового поколения
        self.current_fitness = self.evaluate_population()
        return self.current_fitness

    def run_to_end(self):
        # Запустить до конца, с выводом в консоль можно делать в GUI
        while self.current_generation < self.params['num_generations']:
            self.step()

    def reset(self):
        self.initialize_population()