from PyQt5.QtWidgets import  QApplication
from GUIs.HeatmapGUI import HeatmapGUI
import sys

def main():
    app = QApplication(sys.argv)
    window = HeatmapGUI()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
