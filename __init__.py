# __init__.py

from .plugin import CellNeighborPlannerPlugin

def classFactory(iface):
    """QGIS 插件入口，返回插件主类实例"""
    return CellNeighborPlannerPlugin(iface)