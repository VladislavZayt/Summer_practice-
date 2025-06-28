import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from genetic_algorithm import GeneticAlgorithm

from matplotlib.patches import Circle
from genetic_algorithm import GeneticAlgorithm

class GAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Задача покрытия окружностями")
        self.setGeometry(100, 100, 1200, 600)

        self.canvas = GAVisualizer(self)
        self.init_ui()

        self.points = []
        self.ga = None

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

        self.btn_init = QPushButton("Инициализировать ГА")
        self.btn_init.clicked.connect(self.init_ga)
        layout.addWidget(self.btn_init)

        self.btn_step = QPushButton("Следующий шаг")
        self.btn_step.clicked.connect(self.step_ga)
        layout.addWidget(self.btn_step)

        self.btn_run = QPushButton("До конца")
        self.btn_run.clicked.connect(self.run_to_end)
        layout.addWidget(self.btn_run)

        self.btn_step_through = QPushButton("Пошаговый просмотр не работает")
        self.btn_step_through.clicked.connect(self.step_through_all)
        layout.addWidget(self.btn_step_through)

        layout.addWidget(QLabel("Параметры:"))

        self.param_widgets = {}
        form = QFormLayout()
        param_defs = {
            'pop_size': (10, 200, 20),
            'num_generations': (1, 200, 50),
            'crossover_rate': (0.0, 1.0, 0.7),
            'mutation_rate': (0.0, 1.0, 0.1),
            'penalty_coeff': (0.0, 5.0, 1.0),
            'r_max': (1.0, 50.0, 20.0),
            'sigma': (0.1, 20.0, 5.0),
            'tournament_size': (1, 10, 3),
            'circles_count': (1, 20, 5),
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

        self.btn_clear = QPushButton("Очистить всё")
        self.btn_clear.clicked.connect(self.reset_all)
        self.btn_clear.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        layout.addWidget(self.btn_clear)


    def init_circles(self):
        params = self.get_params()
        count = params['circles_count']
        r_max = params['r_max']
        self.circles = []

        for _ in range(count):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            r = random.uniform(5, r_max)
            self.circles.append((x, y, r))

        self.canvas.set_data(self.points, self.circles)
        
    def get_params(self):
        return {k: w.value() for k, w in self.param_widgets.items()}

    def generate_points(self):
        self.points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(40)]
        self.canvas.set_points(self.points)
        self.canvas.set_data(self.points, self.circles)

    def init_ga(self):
        if not hasattr(self, "points"):
            self.generate_points()
        self.ga = GeneticAlgorithm(self.points, self.get_params())
        self.ga.initialize_population()
        self.canvas.set_ga(self.ga)

    def step_ga(self):
        if self.ga:
            self.ga.step()
            self.canvas.update_plot()
            gen = self.ga.current_generation
            best = self.ga.stats['best'][-1]
            avg = self.ga.stats['average'][-1]
            print(f"Gen {gen}: Best = {best:.2f}, Avg = {avg:.2f}")

    def run_to_end(self):
        if self.ga:
            while self.ga.current_generation < self.ga.params['num_generations']:
                self.ga.step()
                gen = self.ga.current_generation
                best = self.ga.stats['best'][-1]
                avg = self.ga.stats['average'][-1]
                print(f"Gen {gen}: Best = {best:.2f}, Avg = {avg:.2f}")
            self.canvas.update_plot()

    def step_through_all(self):
        print("Пошаговый просмотр не работает")

    def reset_all(self):
        self.points = []


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
                        if not all(isinstance(v, (int, float)) for v in (x, y, r)):
                            print("Invalid circle params:", x, y, r)
                            continue
                        if r < 0:
                            print("Negative radius:", r)
                            continue
                        from matplotlib.patches import Circle
                        circ = Circle((x, y), r, fill=False, edgecolor='red')
                        ax.add_patch(circ)
                except Exception as e:
                    print("Error plotting circles:", e)
            ax2 = self.figure.add_subplot(122)
            ax2.set_title("Fitness by generation")
            if self.ga:
                try:
                    ax2.plot(self.ga.stats['best'], label="Best")
                    ax2.plot(self.ga.stats['average'], label="Average")
                    ax2.legend()
                except Exception as e:
                    print("Error plotting fitness:", e)
            self.canvas.draw()
        except Exception as e:
            print("Exception in update_plot outer:", e)
        print("update_plot: end")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GAWindow()
    window.show()
    sys.exit(app.exec_())
