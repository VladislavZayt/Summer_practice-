import sys
from PyQt5.QtWidgets import QApplication
from example_gui import GAWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GAWindow()
    window.show()
    sys.exit(app.exec_())
