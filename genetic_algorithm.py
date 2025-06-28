import random

class GeneticAlgorithm:
    def __init__(self, points, M, params):
        self.points = points  # список (x, y)
        self.params = params  # dict с параметрами GA
        self.M = params.get("circles_count") # число окружностей
        self.bounding_box = self._compute_bounding_box(points)
        self.population = []  # список особей: каждая особь — list длины 3*M
        self.history = []  # для сохранения поколений (если нужно)
        self.current_generation = 0
        self.stats = {'best': [], 'average': []}
        self.current_fitness = []

    def _compute_bounding_box(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), max(xs), min(ys), max(ys))

    def _has_intersections(self, individual):
        # Проверяет, есть ли пересечения между любыми двумя окружностями в individual
        for i in range(self.M):
            xi, yi, ri = individual[3 * i], individual[3 * i + 1], individual[3 * i + 2]
            for j in range(i + 1, self.M):
                xj, yj, rj = individual[3 * j], individual[3 * j + 1], individual[3 * j + 2]
                dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
                if dist2 < (ri + rj) ** 2:
                    return True
        return False

    def initialize_population(self):
        #Генерирует популяцию из pop_size особей, каждая с M непересекающимися окружностями.    
        self.population = []
        attempts_limit = self.params.get('attempts_limit', 1000)
        xmin, xmax, ymin, ymax = self.bounding_box
        pop_size = self.params['pop_size']

        for idx in range(pop_size):
            created = False
            for attempt in range(attempts_limit):
                individual = []
                circles = []
                valid = True
                for _ in range(self.M):
                    x = random.uniform(xmin, xmax)
                    y = random.uniform(ymin, ymax)
                    r = random.uniform(self.params.get('r_min', 1.0), self.params['r_max'])
                    # Проверка пересечения с уже сгенерированными
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
        if len(self.population) < pop_size and self.population:
            needed = pop_size - len(self.population)
            best = max(self.population, key=self._fitness_no_penalty)
            for _ in range(needed):
                self.population.append(best.copy())
            print(f"Популяция увеличена дубликатами лучшей особи, чтобы достичь pop_size={pop_size}.")

        self.current_generation = 0
        self.history = [list(self.population)]
        self.stats = {'best': [], 'average': []}
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
        # Число покрытых точек для непересекающихся окружностей
        cover_count = 0
        for (px, py) in self.points:
            for i in range(self.M):
                x, y, r = individual[3 * i], individual[3 * i + 1], individual[3 * i + 2]
                if (px - x) ** 2 + (py - y) ** 2 <= r ** 2:
                    cover_count += 1
                    break
        return cover_count

    def select(self):
        # Турнирная селекция
        k = self.params.get('tournament_size', 3)
        pop_len = len(self.population)
        if pop_len == 0:
            raise ValueError("Популяция пуста при селекции!")
        if k > pop_len:
            k = pop_len
        indices = random.sample(range(pop_len), k)
        best_idx = max(indices, key=lambda i: self.current_fitness[i])
        return self.population[best_idx].copy()

    def crossover(self, p1, p2):
        #Одноточечный кроссовер
    
        max_tries = self.params.get('crossover_tries', 5)
        # Функция для попытки кроссовера один раз:
        def make_children(pt1, pt2, point):
            child1 = pt1[:point] + pt2[point:]
            child2 = pt2[:point] + pt1[point:]
            return child1, child2

        for _ in range(max_tries):
            # Выбор точки скрещевания 
            cut_idx = random.randint(1, self.M - 1) * 3
            child1, child2 = make_children(p1, p2, cut_idx)
            # Проверка пересечения 
            ok1 = not self._has_intersections(child1)
            ok2 = not self._has_intersections(child2)
            if ok1 and ok2:
                return child1, child2
            if ok1 and not ok2:
                return child1, p1.copy()
            if ok2 and not ok1:
                return child2, p2.copy()
        return p1.copy(), p2.copy()

    def mutate(self, individual):
        # Вещественное мутирование 
        original = individual.copy()
        xmin, xmax, ymin, ymax = self.bounding_box
        changed = False
        for i in range(self.M):
            if random.random() < self.params.get('mutation_rate', 0.1):
                xi = individual[3 * i] + random.gauss(0, self.params.get('sigma', 1.0))
                yi = individual[3 * i + 1] + random.gauss(0, self.params.get('sigma', 1.0))
                ri = individual[3 * i + 2] + random.gauss(0, self.params.get('sigma', 1.0))

                xi = min(max(xi, xmin), xmax)
                yi = min(max(yi, ymin), ymax)
                ri = min(max(ri, self.params.get('r_min', 1.0)), self.params['r_max'])
                individual[3 * i] = xi
                individual[3 * i + 1] = yi
                individual[3 * i + 2] = ri
                changed = True
        if changed:
            # Проверка пересечений
            if self._has_intersections(individual):
                # откат 
                individual[:] = original[:]

    def step(self):
        # Оценка текущего поколения
        self.current_fitness = self.evaluate_population()

        new_population = []
        pop_size = self.params['pop_size']

        if self.population:
            best_idx = self.current_fitness.index(max(self.current_fitness))
            new_population.append(self.population[best_idx].copy())

        # Генерация остальных особей
        while len(new_population) < pop_size:
            parent1 = self.select()
            parent2 = self.select()
            # Кроссовер
            if random.random() < self.params.get('crossover_rate', 0.7):
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Мутация
            self.mutate(child1)
            if len(new_population) < pop_size:
                new_population.append(child1)
            if len(new_population) < pop_size:
                self.mutate(child2)
                new_population.append(child2)

        # Обновление состояния 
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
