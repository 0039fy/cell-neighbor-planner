# plugin.py
import os
from qgis.PyQt.QtWidgets import QAction, QMenu
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsMessageLog
from qgis.gui import QgisInterface  # 仅用于类型提示

from .main_window import MainWindow
from .constants import STANDARD_FIELDS


class CellNeighborPlannerPlugin:
    """QGIS邻区规划插件"""

    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.toolbar = None
        self.menu = None
        self.action = None
        self.main_window = None

        # 核心数据
        self.target_cells = []
        self.plan_results = []
        self.cell_cache = {
            "4g": {"features": {}, "cell_names": set(), "spatial_index": None, "feature_id_to_name": {}},
            "5g": {"features": {}, "cell_names": set(), "spatial_index": None, "feature_id_to_name": {}}
        }
        self.current_net_type = "4g"
        self.analysis_mode = "5g_4g"
        self.selected_4g_layer = None
        self.selected_5g_layer = None
        self.saved_mapping = {
            "4g": {f: "" for f in STANDARD_FIELDS},
            "5g": {f: "" for f in STANDARD_FIELDS}
        }
        self.config = {
            "macro_max_dist": 2000.0,
            "indoor_max_dist": 800.0,
            "macro_max_neighbors": 64,
            "indoor_max_neighbors": 36,
            "use_azimuth_match": False,
            "distance_weight": 1.0,
            "coverage_weight": 0.0,
            "macro_coverage_range": 1500.0,
            "indoor_coverage_range": 300.0,
            "macro_lobe_width": 65.0,
            "indoor_lobe_width": 90.0,
        }

    def initGui(self):
        """创建菜单与工具栏按钮"""
        self.menu = QMenu("邻区规划工具")
        self.iface.pluginMenu().addMenu(self.menu)

        self.toolbar = self.iface.addToolBar("邻区规划")
        self.toolbar.setObjectName("NeighborPlanningToolbar")

        # 尝试加载图标
        plugin_dir = os.path.dirname(__file__)
        icon_path = os.path.join(plugin_dir, 'icon.png')
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            icon = QIcon()
            QgsMessageLog.logMessage(f"未找到图标: {icon_path}", "邻区规划工具")

        self.action = QAction(icon, "邻区规划", self.iface.mainWindow())
        self.action.triggered.connect(self.show_main_window)
        self.toolbar.addAction(self.action)
        self.menu.addAction(self.action)

    def unload(self):
        """清理插件"""
        if self.menu:
            self.iface.pluginMenu().removeAction(self.menu.menuAction())
            self.menu.deleteLater()
            self.menu = None
        if self.toolbar:
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.main_window:
            self.main_window.close()
            self.main_window.deleteLater()
            self.main_window = None

    def show_main_window(self):
        """显示主窗口（单例）"""
        if not self.main_window:
            self.main_window = MainWindow(self.iface, self)
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()


# QGIS插件入口函数
def classFactory(iface):
    return CellNeighborPlannerPlugin(iface)