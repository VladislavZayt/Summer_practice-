import sys
import copy
import random
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox,
    QInputDialog, QFileDialog, QComboBox, QMessageBox
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from genetic_algorithm import GeneticAlgorithm
from input import load_data


class GAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        import os
        log_path = os.path.abspath("ga_log.txt")
        self.log_file = open(log_path, "w", encoding="utf-8")

        self.setWindowTitle("Задача покрытия окружностями")
        self.setGeometry(100, 100, 1200, 600)

        self.canvas = GAVisualizer(self)
        self.init_ui()
        self.ga = None
        self.current_step = 0
        self.max_steps = 0
        self.crossover_type = 0
        self.mutation_type = 0

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        control_panel = QVBoxLayout()
        self.init_controls(control_panel)
        layout.addLayout(control_panel, 1)

        layout.addWidget(self.canvas, 3)

    def init_controls(self, layout):

        layout.addWidget(QLabel("Расстановка точек:"))
        points_layout = QVBoxLayout()

        self.btn_generate = QPushButton("Генерировать точки")
        self.btn_generate.clicked.connect(self.generate_points)
        points_layout.addWidget(self.btn_generate)

        self.btn_load_file = QPushButton("Загрузить точки из файла")
        self.btn_load_file.clicked.connect(self.load_points_from_file)
        points_layout.addWidget(self.btn_load_file)

        self.btn_manual = QPushButton("Ввести точки вручную")
        self.btn_manual.clicked.connect(self.input_points)
        points_layout.addWidget(self.btn_manual)

        layout.addLayout(points_layout)


        type_layout = QHBoxLayout()

        vbox1 = QVBoxLayout()
        vbox1.addWidget(QLabel("Тип скрещивания:"))
        self.combo_box = QComboBox()
        self.combo_box.addItems([
            "Одноточечное скрещивание", "Двухточечное скрещивание",
            "Равномерное скрещивание", "Скрещивание смещением"
        ])
        self.combo_box.currentIndexChanged.connect(self.on_combo_box_changed)
        vbox1.addWidget(self.combo_box)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(QLabel("Тип мутации:"))
        self.combo_box_2 = QComboBox()
        self.combo_box_2.addItems([
            "Вещественная мутация", "Мутация радиуса", "Мутация слиянием",
            "Мутация равномерным шумом", "Комбинированная мутация"
        ])
        self.combo_box_2.currentIndexChanged.connect(self.on_combo_box_changed_2)
        vbox2.addWidget(self.combo_box_2)

        type_layout.addLayout(vbox1)
        type_layout.addLayout(vbox2)
        layout.addLayout(type_layout)

        layout.addWidget(QLabel("Параметры:"))
        self.param_widgets = {}
        form = QFormLayout()

        param_defs = {
            #"num_points": (1, 50, 20),
            'num_generations': (1, 1000, 50),
            'crossover_rate': (0.0, 1.0, 0.7),
            'mutation_rate': (0.0, 1.0, 0.1),
            'mu': (0.0, 10.0, 0.0),
            'sigma': (0.1, 20.0, 5.0),
            'delta': (1.0, 10.0, 2.0),
            'tournament_size': (1, 10, 3),
            "circles_count": (2, 20, 5),
        }
        param_descriptions = {
            #"num_points": "Количество точек (num_points)",
            'num_generations': "Количество поколений (num_generations)",
            'crossover_rate': "Вероятность скрещивания (crossover_rate)",
            'mutation_rate': "Вероятность мутации (mutation_rate)",
            'mu': "Среднее значение для мутации (mu)",
            'sigma': "Стандартное отклонение для мутации (sigma)",
            'delta': "Дельта для мутации (delta)",
            'tournament_size': "Размер турнира (tournament_size)",
            "circles_count": "Количество окружностей (circles_count)"
        }

        for key, (minv, maxv, default) in param_defs.items():
            if isinstance(default, float):
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setSingleStep(0.1)
            else:
                w = QSpinBox()
            w.setRange(minv, maxv)
            w.setValue(default)
            self.param_widgets[key] = w
            form.addRow(param_descriptions.get(key, key), w)

        layout.addLayout(form)


        self.btn_init = QPushButton("Инициализация")
        self.btn_init.clicked.connect(self.init_ga)
        layout.addWidget(self.btn_init)

        self.btn_clear_circles = QPushButton("Очистить окружности")
        self.btn_clear_circles.clicked.connect(self.clear_circles_only)
        self.btn_clear_circles.setStyleSheet("background-color: orange; color: white; font-weight: bold;")
        layout.addWidget(self.btn_clear_circles)

        self.btn_step = QPushButton("Следующий шаг")
        self.btn_step.clicked.connect(self.step_ga)
        layout.addWidget(self.btn_step)

        self.btn_back = QPushButton("Шаг назад")
        self.btn_back.clicked.connect(self.step_back)
        layout.addWidget(self.btn_back)

        self.btn_step_through = QPushButton("Пошаговый просмотр")
        self.btn_step_through.clicked.connect(self.step_through_all)
        layout.addWidget(self.btn_step_through)

        self.btn_run = QPushButton("До конца")
        self.btn_run.clicked.connect(self.run_to_end)
        layout.addWidget(self.btn_run)

        self.btn_clear = QPushButton("Очистить всё")
        self.btn_clear.clicked.connect(self.reset_all)
        self.btn_clear.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        layout.addWidget(self.btn_clear)

    def on_combo_box_changed(self, index):
        self.crossover_type = index

    def on_combo_box_changed_2(self, index):
        self.mutation_type = index

    def clear_circles_only(self):
        if hasattr(self, 'ga'):
            self.ga = None
            self.canvas.reset()

    def generate_points(self):
        num_points, ok = QInputDialog.getInt(self, "Генерация точек", "Введите количество точек:", min=1, max=200)
        if not ok:
            return
        self.points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]
        self.canvas.set_points(self.points)

    def load_points_from_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt);;All Files (*)")
        if not filename:
            return
        n, _ = QInputDialog.getInt(self, " ", "Сколько точек загрузить?", min=1)
        self.points = load_data(filename, n)
        self.canvas.set_points(self.points)
        #self.param_widgets["num_points"].setValue(n)

    def input_points(self):
        if not hasattr(self, 'points'):
            self.points = []

        MAX_COORD = 1000000

        while True:
            text, ok = QInputDialog.getText(
                self,
                "Ввод точки",
                "Введите координаты X Y или X,Y (оставьте пустым для завершения):"
            )
            if not ok or not text.strip():
                break

            try:
                parts = text.replace(',', ' ').split()
                if len(parts) != 2:
                    raise ValueError("Нужно ввести две координаты.")

                x, y = float(parts[0]), float(parts[1])
                if abs(x) > MAX_COORD or abs(y) > MAX_COORD:
                    raise ValueError(f"Координаты не должны превышать {MAX_COORD}.")

                self.points.append((x, y))
                self.canvas.set_points(self.points)
                #self.param_widgets["num_points"].setValue(len(self.points))

            except ValueError as ve:
                QMessageBox.warning(self, "Ошибка ввода", str(ve))
            except Exception:
                QMessageBox.warning(
                    self, "Ошибка ввода",
                    "Неверный формат. Введите как: 3.14 36.6 или 36.6,3.14"
                )

    def get_params(self):
        params = {k: w.value() for k, w in self.param_widgets.items()}
        params["crossover_type"] = self.crossover_type
        params["mutation_type"] = self.mutation_type
        return params

    def init_ga(self):
        if not hasattr(self, "points") or not self.points:
            self.generate_points()
        self.ga = GeneticAlgorithm(self.points, self.get_params())
        self.ga.initialize_population()
        self.ga.history = [copy.deepcopy(self.ga.population)]
        self.canvas.set_ga(self.ga)

    def step_ga(self):
        if self.ga:
            self.ga.step()
            self.ga.history.append(copy.deepcopy(self.ga.population))
            self.canvas.update_plot()
            gen = self.ga.current_generation
            best = self.ga.stats['best'][-1]
            avg = self.ga.stats['average'][-1]
            self.print_generation_stats()

    def run_to_end(self):
        if self.ga:
            try:
                while self.ga.current_generation < self.ga.params['num_generations']:
                    self.ga.step()
                    self.ga.history.append(copy.deepcopy(self.ga.population))
                self.canvas.update_plot()
                self.print_generation_stats()
                self.ga.save_best_solution("best_solution.txt")
            except Exception as e:
                print("Ошибка при выполнении до конца:", e)

    def step_back(self):
        if self.ga and self.ga.current_generation > 0 and len(self.ga.history) > 1:
            self.ga.current_generation -= 1
            self.ga.population = copy.deepcopy(self.ga.history[self.ga.current_generation])
            self.ga.history = self.ga.history[:self.ga.current_generation + 1]
            self.ga.evaluate_population()
            self.canvas.update_plot()
            self.print_generation_stats()
            print(f"Возврат к поколению {self.ga.current_generation}")
        else:
            print("Нельзя вернуться назад — это первое поколение")

    def step_through_all(self):
        if not self.ga:
            return

        self.current_step = 0
        self.max_steps = self.ga.params['num_generations']
        self.step_through_tick()

    def step_through_tick(self):
        if not self.ga or self.current_step >= self.max_steps:
            return

        self.ga.step()
        self.canvas.update_plot()
        gen = self.ga.current_generation
        best = self.ga.stats['best'][-1]
        avg = self.ga.stats['average'][-1]
        self.print_generation_stats()

        self.current_step += 1
        QTimer.singleShot(300, self.step_through_tick)

    def print_generation_stats(self):
        if self.ga:
            gen = self.ga.current_generation
            best = self.ga.stats['best'][-1]
            avg = self.ga.stats['average'][-1]
            log_line = f"Gen {gen}: Best = {best:.2f}, Avg = {avg:.2f}\nupdate_plot: start\nupdate_plot: end\n"
            print(log_line, end="")
            self.log_file.write(log_line)
            self.log_file.flush()

    def reset_all(self):
        self.points = []
        self.ga = None
        self.canvas.reset(clear_points=True)

    def closeEvent(self, event):
        #print("closeEvent called")
        if hasattr(self, 'log_file') and not self.log_file.closed:
            #print("Closing log file")
            self.log_file.close()
        event.accept()


class GAVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ga = None
        self.points = []

    def set_points(self, points):
        self.points = points
        self.update_plot()

    def set_ga(self, ga):
        self.ga = ga
        self.update_plot()

    def update_plot(self):
        print("update_plot: start")
        try:
            self.figure.clear()

            import matplotlib.gridspec as gridspec
            gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 2])

            # верхний график
            ax_fitness = self.figure.add_subplot(gs[0])
            ax_fitness.set_title("Лучшее и среднее покрытие", fontsize=14)
            ax_fitness.set_xlabel("Поколение", fontsize=16)
            ax_fitness.set_ylabel("Fitness", fontsize=16)
            ax_fitness.tick_params(axis='both', labelsize=14)

            if self.ga:
                try:
                    best = self.ga.stats['best']
                    avg = self.ga.stats['average']
                    ax_fitness.plot(best, label="Best")
                    ax_fitness.plot(avg, label="Average")
                    ax_fitness.legend(fontsize=14)

                    # Сплющить график
                    max_y = max(max(best), max(avg)) if best and avg else 1
                    ax_fitness.set_ylim(0, max_y * 1.2)

                except Exception as e:
                    print("Ошибка построения графика fitness:", e)

            # нижний график
            ax_cov = self.figure.add_subplot(gs[1])
            ax_cov.set_title("Покрытие", fontsize=16)
            ax_cov.set_xlabel("X координата", fontsize=16)
            ax_cov.set_ylabel("Y координата", fontsize=16)
            ax_cov.tick_params(axis='both', labelsize=14)
            ax_cov.set_aspect('equal')

            if self.points:
                try:
                    xs, ys = zip(*self.points)
                    ax_cov.scatter(xs, ys, color='black')
                except Exception as e:
                    print("Ошибка построения точек:", e)

            if self.ga and self.ga.population:
                try:
                    best = max(self.ga.population, key=self.ga._fitness_no_penalty)
                    for i in range(self.ga.M):
                        x, y, r = best[3 * i], best[3 * i + 1], best[3 * i + 2]
                        if not all(isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v) for v in
                                   (x, y, r)):
                            print("Некорректные параметры окружностей:", x, y, r)
                            continue
                        if r < 0:
                            continue
                        circ = Circle((x, y), r, fill=False, edgecolor='red')
                        ax_cov.add_patch(circ)
                except Exception as e:
                    print("Ошибка построения окружностей:", e)

            self.canvas.draw()
        except Exception as e:
            print("Ошибка изменения графиков:", e)
        print("update_plot: end")

    def reset(self, clear_points=False):
        if clear_points:
            self.points = []
        self.ga = None
        self.update_plot()