import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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

        self.btn_init = QPushButton("Окружности")
        self.btn_init.clicked.connect(self.init_circles)
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
        self.canvas.set_data(self.points, self.circles)

    def init_ga(self):
        if not hasattr(self, "points"):
            self.generate_points()
        self.ga = GeneticAlgorithm(self.points, self.get_params())
        self.ga.initialize_population()
        self.canvas.set_ga(self.ga)

    def step_ga(self):
        print("Следующий шаг не работает")

    def run_to_end(self):
        print("До конца не работает")

    def step_through_all(self):
        print("Пошаговый просмотр не работает")

    def reset_all(self):
        self.points = []
        self.circles = []
        self.canvas.set_data([], [])


class GAVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.points = []
        self.circles = []

    def set_data(self, points, circles):
        self.points = points
        self.circles = circles
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("Покрытие")
        ax.set_aspect('equal')

        if self.points:
            xs, ys = zip(*self.points)
            ax.scatter(xs, ys, color='black')

        for x, y, r in self.circles:
            circ = Circle((x, y), r, fill=False, edgecolor='red')
            ax.add_patch(circ)

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GAWindow()
    window.show()
    sys.exit(app.exec_())
