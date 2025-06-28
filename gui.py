import sys
import copy
import random
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from genetic_algorithm import GeneticAlgorithm


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
        self.btn_generate = QPushButton("Генерировать точки")
        self.btn_generate.clicked.connect(self.generate_points)
        layout.addWidget(self.btn_generate)

        self.btn_init = QPushButton("Инициализация")
        self.btn_init.clicked.connect(self.init_ga)
        layout.addWidget(self.btn_init)

        self.btn_step = QPushButton("Следующий шаг")
        self.btn_step.clicked.connect(self.step_ga)
        layout.addWidget(self.btn_step)

        self.btn_back = QPushButton("Шаг назад")
        self.btn_back.clicked.connect(self.step_back)
        layout.addWidget(self.btn_back)

        self.btn_run = QPushButton("До конца")
        self.btn_run.clicked.connect(self.run_to_end)
        layout.addWidget(self.btn_run)

        self.btn_step_through = QPushButton("Пошаговый просмотр")
        self.btn_step_through.clicked.connect(self.step_through_all)
        layout.addWidget(self.btn_step_through)

        layout.addWidget(QLabel("Параметры:"))

        self.param_widgets = {}
        form = QFormLayout()
        param_defs = {
            "num_points": (1, 50, 20),
            'num_generations': (1, 1000, 50),
            'crossover_rate': (0.0, 1.0, 0.7),
            'mutation_rate': (0.0, 1.0, 0.1),
            'penalty_coeff': (0.0, 5.0, 1.0),
            'sigma': (0.1, 20.0, 5.0),
            'delta': (1.0, 10.0, 2.0),
            'tournament_size': (1, 10, 3),
            "circles_count": (2, 20, 5),
            "crossover_type": (0, 3, 0),
            "mutation_type": (0, 4, 0),
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
            form.addRow(key, w)

        layout.addLayout(form)


        self.btn_clear_circles = QPushButton("Очистить окружности")
        self.btn_clear_circles.clicked.connect(self.clear_circles_only)
        self.btn_clear_circles.setStyleSheet("background-color: orange; color: white; font-weight: bold;")
        layout.addWidget(self.btn_clear_circles)

        self.btn_clear = QPushButton("Очистить всё")
        self.btn_clear.clicked.connect(self.reset_all)
        self.btn_clear.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        layout.addWidget(self.btn_clear)

    def clear_circles_only(self):
        if hasattr(self, 'ga'):
            self.ga = None
            self.canvas.reset()

    def generate_points(self):
        params = self.get_params()
        num_points = params['num_points']
        self.points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]
        self.canvas.set_points(self.points)

    def get_params(self):
        return {k: w.value() for k, w in self.param_widgets.items()}

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
        print("closeEvent called")
        if hasattr(self, 'log_file') and not self.log_file.closed:
            print("Closing log file")
            self.log_file.close()
        event.accept()


class GAVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
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
            ax = self.figure.add_subplot(121)
            ax.set_title("Покрытие")
            ax.set_aspect('equal')
            if self.points:
                try:
                    xs, ys = zip(*self.points)
                    ax.scatter(xs, ys, color='black')
                except Exception as e:
                    print("Error plotting points:", e)

            if self.ga and self.ga.population:
                try:
                    best = max(self.ga.population, key=self.ga._fitness_no_penalty)
                    for i in range(self.ga.M):
                        x, y, r = best[3 * i], best[3 * i + 1], best[3 * i + 2]
                        if not all(isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v) for v in
                                   (x, y, r)):
                            print("Invalid circle params:", x, y, r)
                            continue
                        if r < 0:
                            continue
                        circ = Circle((x, y), r, fill=False, edgecolor='red')
                        ax.add_patch(circ)
                except Exception as e:
                    print("Error plotting circles:", e)

            ax2 = self.figure.add_subplot(122)
            ax2.set_title("Fitness по поколениям")
            if self.ga:
                try:
                    ax2.plot(self.ga.stats['best'], label="Best")
                    ax2.plot(self.ga.stats['average'], label="Average")
                    ax2.legend()
                except Exception as e:
                    print("Error plotting fitness:", e)

            try:
                self.figure.savefig("full_plot.png", dpi=150)

                # Левый график
                coverage_fig = Figure()
                ax_cov = coverage_fig.add_subplot(111)
                ax_cov.set_title("Покрытие")
                ax_cov.set_aspect('equal')
                if self.points:
                    xs, ys = zip(*self.points)
                    ax_cov.scatter(xs, ys, color='black')
                if self.ga and self.ga.population:
                    best = max(self.ga.population, key=self.ga._fitness_no_penalty)
                    for i in range(self.ga.M):
                        x, y, r = best[3 * i], best[3 * i + 1], best[3 * i + 2]
                        circ = Circle((x, y), r, fill=False, edgecolor='red')
                        ax_cov.add_patch(circ)
                coverage_fig.savefig("coverage.png", dpi=150)

                fitness_fig = Figure()
                ax_fit = fitness_fig.add_subplot(111)
                ax_fit.set_title("Fitness by generation")
                if self.ga:
                    ax_fit.plot(self.ga.stats['best'], label="Best")
                    ax_fit.plot(self.ga.stats['average'], label="Average")
                    ax_fit.legend()
                fitness_fig.savefig("fitness.png", dpi=150)

            except Exception as e:
                print("Error while saving figures:", e)

            self.canvas.draw()
        except Exception as e:
            print("Exception in update_plot outer:", e)
        print("update_plot: end")

    def reset(self, clear_points=False):
        if clear_points:
            self.points = []
        self.ga = None
        self.update_plot()