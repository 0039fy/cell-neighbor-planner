# spatial_calculator.py
import math
import numpy as np
from qgis.core import QgsPointXY, QgsRectangle
from qgis.PyQt.QtCore import QObject

class QgisSpatialCalculator(QObject):
    """高性能空间计算工具：距离与方位角均提供向量化批量版本"""

    def __init__(self):
        super().__init__()
        # 保留椭球体对象仅在必要时使用，默认所有计算用快速 Haversine
        from qgis.core import QgsDistanceArea
        self.da = QgsDistanceArea()
        self.da.setEllipsoid('WGS84')

    # ---------- 单点计算（兼容原接口） ----------
    def calculate_distance(self, p1: QgsPointXY, p2: QgsPointXY) -> float:
        return self._haversine(p1.x(), p1.y(), p2.x(), p2.y())

    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371000
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    @staticmethod
    def batch_haversine(lon0, lat0, lons, lats):
        """向量化 Haversine 距离（米）。lons, lats 为 numpy 数组"""
        R = 6371000
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)
        lats = np.radians(lats)
        lons = np.radians(lons)
        dlat = lats - lat0
        dlon = lons - lon0
        a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lats)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    @staticmethod
    def batch_bearing(lon0, lat0, lons, lats):
        """向量化计算从 (lon0,lat0) 到 (lons,lats) 的方位角（度），返回 [0,360)"""
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)
        lats = np.radians(lats)
        lons = np.radians(lons)
        dlon = lons - lon0
        x = np.sin(dlon) * np.cos(lats)
        y = np.cos(lat0) * np.sin(lats) - np.sin(lat0) * np.cos(lats) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(x, y))
        return bearing % 360

    def create_buffered_rectangle(self, center: QgsPointXY, distance_meters: float) -> QgsRectangle:
        lat = center.y()
        delta_lat = distance_meters / 111000.0
        delta_lon = distance_meters / (111000.0 * math.cos(math.radians(lat)))
        return QgsRectangle(
            center.x() - delta_lon, center.y() - delta_lat,
            center.x() + delta_lon, center.y() + delta_lat
        )

    # 保留原 calculate_bearing 以便逐点调用（但批处理版本效率更高）
    def calculate_bearing(self, p1: QgsPointXY, p2: QgsPointXY) -> float:
        return self._bearing_single(p1.x(), p1.y(), p2.x(), p2.y())

    @staticmethod
    def _bearing_single(lon1, lat1, lon2, lat2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360