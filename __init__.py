def classFactory(iface):
    from .neighbor_planning_tool import CellNeighborPlannerPlugin
    return CellNeighborPlannerPlugin(iface)


