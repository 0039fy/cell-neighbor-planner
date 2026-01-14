import math
import time
import numpy as np
import re
from datetime import datetime

# Qt和QGIS模块
from qgis.PyQt.QtWidgets import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsField, QgsFields, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsDistanceArea, QgsSpatialIndex, QgsMessageLog, QgsCoordinateTransform,
    QgsRectangle, QgsFeatureRequest, QgsCoordinateTransformContext,
    QgsWkbTypes, QgsPoint
)
from qgis.gui import QgsMessageBar

# ==================== 全局常量定义 ====================

STANDARD_FIELDS = {
    "基站ID": {"required": True, "patterns": [r'基站ID', r'enodeb.?id', r'gnb.?id', r'基站id']},
    "小区ID": {"required": True, "patterns": [r'小区ID', r'小区编号', r'cell.?id', r'cellLocalId', r'cellid']},
    "小区名称": {"required": True, "patterns": [r'小区名称', r'小区名', r'cell.?name', r'小区中文名']},
    "覆盖类型": {"required": True, "patterns": [r'覆盖类型', r'基站类型', r'覆盖场景', r'cell.?type']},
    "频点": {"required": True, "patterns": [r'频点', r'频段', r'frequency', r'freq', r'earfcn', r'nrarfcn']},
    "经度": {"required": True, "patterns": [r'经度', r'lon', r'LONB', r'经度（\*）']},
    "纬度": {"required": True, "patterns": [r'纬度', r'纬度（*）', r'纬度（\*）', r'lat', r'LATB']},
    "方位角": {"required": True, "patterns": [r'方位角', r'方向角', r'azimuth', r'angle']},
    "子网ID": {"required": False, "patterns": [r'子网ID', r'子网', r'子网编号', r'subnet.?id', r'subnetno']},
    "网元ID": {"required": False, "patterns": [r'网元ID', r'管理网元ID', r'gNBId', r'ne.?id', r'网元标识']},
    "PCI": {"required": False, "patterns": [r'PCI', r'物理小区标识']},
    "TAC": {"required": False, "patterns": [r'TAC', r'跟踪区']},
}

EXPORT_COLUMNS = [
    "源网络类型", "目标网络类型", "源子网ID", "源网元ID", "源基站ID", "源小区ID", "源小区名称",
    "源覆盖类型", "源频点", "源方位角", "目标子网ID", "目标网元ID", "目标基站ID", "目标小区ID",
    "目标小区名称", "目标覆盖类型", "目标频点", "目标方位角", "距离(m)",
    "覆盖相关度", "方位角匹配度", "综合得分", "邻区类型", "规划时间"
]


# ==================== 工具函数 ====================

def normalize_angle_diff(angle1, angle2):
    """标准化角度差"""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def normalize_string(s):
    """标准化字符串"""
    if s is None:
        return ""
    return str(s).strip()


# ==================== QGIS空间计算类 ====================

class QgisSpatialCalculator:
    """QGIS空间计算工具类"""

    def __init__(self):
        self.distance_area = QgsDistanceArea()
        self.distance_area.setEllipsoid('WGS84')

    def calculate_distance(self, point1, point2):
        """使用QGIS计算距离（考虑椭球体）"""
        return self.distance_area.measureLine([point1, point2])

    def create_buffered_rectangle(self, center_point, distance_meters):
        """创建缓冲矩形（用于空间查询）"""
        # 将距离转换为度（近似值）
        # 1度纬度约111km，经度在赤道约111km，在纬度φ处为111km * cos(φ)
        lat = center_point.y()
        delta_lat = distance_meters / 111000.0
        delta_lon = distance_meters / (111000.0 * math.cos(math.radians(lat)))

        return QgsRectangle(
            center_point.x() - delta_lon,
            center_point.y() - delta_lat,
            center_point.x() + delta_lon,
            center_point.y() + delta_lat
        )

    def calculate_bearing(self, point1, point2):
        """计算方位角"""
        # 使用Haversine公式计算方位角
        lat1 = math.radians(point1.y())
        lon1 = math.radians(point1.x())
        lat2 = math.radians(point2.y())
        lon2 = math.radians(point2.x())

        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing


# ==================== 规划线程类 ====================

class PlanningThread(QThread):
    progress_update = pyqtSignal(int, str)
    result_update = pyqtSignal(list)
    finish_signal = pyqtSignal(dict, float)
    error_signal = pyqtSignal(str)
    failure_report = pyqtSignal(list)

    def __init__(self, target_cells, mode, config, cell_cache, saved_mapping):
        super().__init__()
        self.target_cells = target_cells
        self.mode = mode
        self.config = config
        self.cell_cache = cell_cache
        self.saved_mapping = saved_mapping

        # 进度控制
        total_cells = len(target_cells)
        self.progress_step = 5 if total_cells < 100 else 10 if total_cells < 1000 else 50
        self.last_progress = 0
        self.result_buffer = []
        self.result_step = 20

        # QGIS空间计算器
        self.spatial_calc = QgisSpatialCalculator()

        # 失败清单
        self.failure_list = []

    def run(self):
        """运行规划"""
        try:
            all_results = []
            total = len(self.target_cells)
            stats = {
                "total_processed": 0,
                "success_count": 0,
                "failed_count": 0,
                "total_neighbors": 0,
                "avg_score": 0.0,
                "plan_time": 0.0
            }

            start_time = time.time()

            for idx, cell_input in enumerate(self.target_cells):
                current_idx = idx + 1

                # 更新进度
                progress = int(current_idx / total * 100)
                update_condition = (
                        current_idx % self.progress_step == 0 or
                        current_idx == total or
                        progress - self.last_progress >= 1
                )

                if update_condition:
                    msg = f"处理中: {cell_input} ({current_idx}/{total})"
                    self.progress_update.emit(progress, msg)
                    self.last_progress = progress

                # 规划单个小区
                results, failure_reason = self.plan_single_cell_with_reason(cell_input, self.mode)
                stats["total_processed"] += 1

                if results:
                    all_results.extend(results)
                    stats["success_count"] += 1
                    stats["total_neighbors"] += len(results)

                    self.result_buffer.extend(results)
                    if len(self.result_buffer) >= self.result_step or current_idx == total:
                        self.result_update.emit(self.result_buffer)
                        self.result_buffer = []
                else:
                    stats["failed_count"] += 1
                    # 记录失败原因
                    if failure_reason:
                        self.failure_list.append({
                            "小区": cell_input,
                            "失败原因": failure_reason,
                            "失败时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            # 最终进度
            self.progress_update.emit(100, f"处理完成：共处理{total}个小区")

            stats["plan_time"] = time.time() - start_time
            if all_results:
                scores = [r["综合得分"] for r in all_results if isinstance(r.get("综合得分"), (int, float))]
                if scores:
                    stats["avg_score"] = np.mean(scores)

            # 发送失败报告
            if self.failure_list:
                self.failure_report.emit(self.failure_list)

            self.finish_signal.emit(stats, stats["plan_time"])

        except Exception as e:
            self.error_signal.emit(str(e))

    def plan_single_cell_with_reason(self, target_input, mode):
        """规划单个小区，返回结果和失败原因"""
        # 确定网络类型
        if mode == "5g_4g":
            source_net_type, target_net_type = "5g", "4g"
        elif mode == "5g_only":
            source_net_type = target_net_type = "5g"
        elif mode == "4g_only":
            source_net_type = target_net_type = "4g"
        else:
            return [], "未知规划模式"

        # 解析目标小区
        matched_cell = self.parse_target_cell(target_input, source_net_type)
        if not matched_cell:
            return [], f"源小区'{target_input}'在{source_net_type.upper()}图层中不存在"

        # 获取源小区信息
        source_cache = self.cell_cache[source_net_type]
        if matched_cell not in source_cache["features"]:
            return [], f"源小区'{matched_cell}'在{source_net_type.upper()}缓存中不存在"

        source_feature = source_cache["features"][matched_cell]
        source_point = source_feature["point"]
        source_type = source_feature["type"]
        source_azimuth = source_feature["azimuth"]

        # 获取距离限制
        if source_type == "宏站":
            max_dist = self.config["macro_max_dist"]
            max_neighbors = self.config["macro_max_neighbors"]
        else:
            max_dist = self.config["indoor_max_dist"]
            max_neighbors = self.config["indoor_max_neighbors"]

        # 获取目标缓存
        target_cache = self.cell_cache[target_net_type]

        # 使用空间索引快速查找候选小区
        candidate_cells = []

        # 创建缓冲矩形
        rect = self.spatial_calc.create_buffered_rectangle(source_point, max_dist)

        # 如果有空间索引，使用空间索引查询
        if target_cache.get("spatial_index"):
            spatial_index = target_cache["spatial_index"]
            candidate_ids = spatial_index.intersects(rect)

            for feature_id in candidate_ids:
                cell_name = target_cache["feature_id_to_name"].get(feature_id)
                if not cell_name or (cell_name == matched_cell and source_net_type == target_net_type):
                    continue

                target_feature = target_cache["features"].get(cell_name)
                if target_feature:
                    target_point = target_feature["point"]
                    distance = self.spatial_calc.calculate_distance(source_point, target_point)

                    if distance <= max_dist:
                        candidate_cells.append({
                            "cell_name": cell_name,
                            "feature": target_feature,
                            "distance": distance
                        })
        else:
            # 如果没有空间索引，遍历所有要素
            for cell_name, target_feature in target_cache["features"].items():
                # 跳过自身
                if cell_name == matched_cell and source_net_type == target_net_type:
                    continue

                # 计算距离
                target_point = target_feature["point"]
                distance = self.spatial_calc.calculate_distance(source_point, target_point)

                if distance <= max_dist:
                    candidate_cells.append({
                        "cell_name": cell_name,
                        "feature": target_feature,
                        "distance": distance
                    })

        if not candidate_cells:
            return [], f"指定距离内({max_dist}米)无可用邻区"

        # 计算得分并排序
        scored_candidates = []
        for candidate in candidate_cells:
            score_result = self.calculate_comprehensive_score(
                source_feature, candidate["feature"],
                source_net_type, target_net_type,
                candidate["distance"]
            )

            scored_candidates.append({
                "cell_name": candidate["cell_name"],
                "feature": candidate["feature"],
                "distance": candidate["distance"],
                "total_score": score_result["total_score"],
                "components": score_result["components"],
                "neighbor_type": score_result["neighbor_type"]
            })

        # 排序并选择前N个
        scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        selected_candidates = scored_candidates[:max_neighbors]

        # 构建结果
        final_results = []
        for candidate in selected_candidates:
            result_data = self.build_result_data(
                source_feature, candidate["feature"],
                source_net_type, target_net_type, candidate
            )
            final_results.append(result_data)

        return final_results, None  # 成功，没有失败原因

    def calculate_comprehensive_score(self, source_feature, target_feature,
                                      source_net_type, target_net_type, distance):
        """计算综合得分"""
        source_type = source_feature["type"]
        target_type = target_feature["type"]
        source_azimuth = source_feature["azimuth"]
        target_azimuth = target_feature["azimuth"]

        use_azimuth = self.config["use_azimuth_match"]

        if not use_azimuth:
            # 纯距离模式
            distance_score = self.calculate_distance_score(distance, source_type, target_type)
            total_score = distance_score
            coverage_score = 0.0
            azimuth_match_score = 0.0
        else:
            # 方位角匹配模式
            distance_score = self.calculate_distance_score(distance, source_type, target_type)
            coverage_score = self.calculate_coverage_score(
                source_feature, target_feature, distance
            )
            azimuth_match_score = self.calculate_azimuth_match_score(
                source_feature, target_feature, distance
            )

            distance_weight = self.config["distance_weight"]
            coverage_weight = self.config["coverage_weight"]

            total_score = (
                    distance_score * distance_weight +
                    coverage_score * coverage_weight +
                    azimuth_match_score * coverage_weight
            )

        # 同站加分
        same_site_bonus = 0.0
        if (source_feature["station_id"] and target_feature["station_id"] and
                source_feature["station_id"] == target_feature["station_id"] and
                distance <= 100):
            same_site_bonus = 0.2
            total_score += same_site_bonus

        total_score = max(0.0, min(1.0, total_score))
        neighbor_type = self.determine_neighbor_type(distance, total_score, use_azimuth)

        return {
            "total_score": total_score,
            "components": {
                "distance_score": distance_score,
                "coverage_score": coverage_score,
                "azimuth_match_score": azimuth_match_score,
                "same_site_bonus": same_site_bonus
            },
            "neighbor_type": neighbor_type
        }

    def calculate_distance_score(self, distance, source_type, target_type):
        """计算距离得分"""
        if source_type == "宏站" and target_type == "宏站":
            max_dist = self.config["macro_max_dist"]
        else:
            max_dist = self.config["indoor_max_dist"]

        if distance <= 50:
            return 1.0
        elif distance <= max_dist:
            normalized_dist = (distance - 50) / (max_dist - 50)
            return math.exp(-0.7 * normalized_dist)
        else:
            return 0.0

    def calculate_coverage_score(self, source_feature, target_feature, distance):
        """计算覆盖相关度"""
        # 获取方位角
        source_azimuth = source_feature["azimuth"]
        target_azimuth = target_feature["azimuth"]

        # 计算方位角差
        bearing_st = self.spatial_calc.calculate_bearing(
            source_feature["point"], target_feature["point"]
        )
        bearing_ts = self.spatial_calc.calculate_bearing(
            target_feature["point"], source_feature["point"]
        )

        angle_diff_st = normalize_angle_diff(bearing_st, source_azimuth)
        angle_diff_ts = normalize_angle_diff(bearing_ts, target_azimuth)

        # 获取覆盖参数
        def get_coverage_params(cell_type):
            if cell_type == "宏站":
                return {
                    "range": self.config["macro_coverage_range"],
                    "lobe_width": self.config["macro_lobe_width"]
                }
            else:
                return {
                    "range": self.config["indoor_coverage_range"],
                    "lobe_width": self.config["indoor_lobe_width"]
                }

        source_params = get_coverage_params(source_feature["type"])
        target_params = get_coverage_params(target_feature["type"])

        # 计算覆盖概率
        def calculate_coverage_probability(distance, angle_diff, coverage_range, lobe_width):
            if distance <= coverage_range:
                distance_factor = math.exp(-(distance / coverage_range) ** 2)
            else:
                distance_factor = max(0, 1 - (distance - coverage_range) / coverage_range)

            half_lobe = lobe_width / 2
            if angle_diff <= half_lobe:
                angle_factor = 1.0 - (angle_diff / half_lobe) * 0.4
            elif angle_diff <= half_lobe * 2:
                angle_factor = 0.6 - ((angle_diff - half_lobe) / half_lobe) * 0.4
            else:
                angle_factor = max(0.0, 0.2 - ((angle_diff - half_lobe * 2) / 180) * 0.2)

            return distance_factor * angle_factor

        coverage_st = calculate_coverage_probability(
            distance, angle_diff_st,
            source_params["range"], source_params["lobe_width"]
        )

        coverage_ts = calculate_coverage_probability(
            distance, angle_diff_ts,
            target_params["range"], target_params["lobe_width"]
        )

        coverage_score = math.sqrt(coverage_st * coverage_ts)
        return min(1.0, max(0.0, coverage_score))

    def calculate_azimuth_match_score(self, source_feature, target_feature, distance):
        """计算方位角匹配度"""
        source_azimuth = source_feature["azimuth"]
        target_azimuth = target_feature["azimuth"]

        if source_azimuth == 0.0 or target_azimuth == 0.0:
            return 0.5

        bearing_st = self.spatial_calc.calculate_bearing(
            source_feature["point"], target_feature["point"]
        )
        bearing_ts = self.spatial_calc.calculate_bearing(
            target_feature["point"], source_feature["point"]
        )

        angle_diff_st = normalize_angle_diff(bearing_st, source_azimuth)
        angle_diff_ts = normalize_angle_diff(bearing_ts, target_azimuth)

        def single_match_score(angle_diff):
            if angle_diff <= 30:
                return 1.0 - (angle_diff / 30) * 0.3
            elif angle_diff <= 60:
                return 0.7 - ((angle_diff - 30) / 30) * 0.3
            elif angle_diff <= 90:
                return 0.4 - ((angle_diff - 60) / 30) * 0.2
            elif angle_diff <= 120:
                return 0.2 - ((angle_diff - 90) / 30) * 0.1
            else:
                return max(0.0, 0.1 - ((angle_diff - 120) / 60) * 0.1)

        match_st = single_match_score(angle_diff_st)
        match_ts = single_match_score(angle_diff_ts)

        match_score = math.sqrt(match_st * match_ts)
        return match_score

    def determine_neighbor_type(self, distance, score, use_azimuth):
        """确定邻区类型"""
        if distance <= 500:
            base_type = "近距离"
        elif distance <= 1500:
            base_type = "中距离"
        else:
            base_type = "远距离"

        if score >= 0.8:
            relevance = "强相关"
        elif score >= 0.6:
            relevance = "中等相关"
        elif score >= 0.4:
            relevance = "弱相关"
        else:
            relevance = "极弱相关"

        mode_tag = "方位角匹配" if use_azimuth else "纯距离"
        return f"{base_type}-{relevance}-{mode_tag}"

    def build_result_data(self, source_feature, target_feature, source_net_type, target_net_type, candidate):
        """构建结果数据"""
        result_data = {
            "源网络类型": "5G" if source_net_type == "5g" else "4G",
            "目标网络类型": "5G" if target_net_type == "5g" else "4G",
            "源子网ID": source_feature.get("subnet_id", ""),
            "源网元ID": source_feature.get("ne_id", ""),
            "源基站ID": source_feature.get("station_id", ""),
            "源小区ID": source_feature.get("cell_id", ""),
            "源小区名称": source_feature.get("cell_name", ""),
            "源覆盖类型": source_feature.get("type", ""),
            "源频点": source_feature.get("frequency", ""),
            "源方位角": source_feature.get("azimuth", ""),
            "目标子网ID": target_feature.get("subnet_id", ""),
            "目标网元ID": target_feature.get("ne_id", ""),
            "目标基站ID": target_feature.get("station_id", ""),
            "目标小区ID": target_feature.get("cell_id", ""),
            "目标小区名称": target_feature.get("cell_name", ""),
            "目标覆盖类型": target_feature.get("type", ""),
            "目标频点": target_feature.get("frequency", ""),
            "目标方位角": target_feature.get("azimuth", ""),
            "距离(m)": round(candidate["distance"], 2),
            "覆盖相关度": round(candidate["components"].get("coverage_score", 0), 4),
            "方位角匹配度": round(candidate["components"].get("azimuth_match_score", 0), 4),
            "综合得分": round(candidate["total_score"], 4),
            "邻区类型": candidate["neighbor_type"],
            "规划时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return result_data

    def parse_target_cell(self, cell_input, net_type):
        """解析目标小区"""
        cell_input = cell_input.strip()
        if not cell_input:
            return None

        cache = self.cell_cache[net_type]

        # 直接查找
        if cell_input in cache["features"]:
            return cell_input

        # 查找ID组合
        if '-' in cell_input:
            for cell_name, feature in cache["features"].items():
                station_id = feature.get("station_id", "")
                cell_id = feature.get("cell_id", "")
                if station_id and cell_id:
                    id_key = f"{station_id}-{cell_id}"
                    if cell_input == id_key:
                        return cell_name

        # 模糊查找
        for cell_name in cache["features"].keys():
            if cell_input in cell_name:
                return cell_name

        return None


# ==================== 主窗口类 ====================

class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, iface, plugin):
        super().__init__()
        self.iface = iface
        self.plugin = plugin

        # 设置窗口属性
        self.setWindowTitle("邻区规划工具 v3.0 - QGIS集成版")
        self.setMinimumSize(1000, 700)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 创建各选项卡
        self.create_layer_tab()
        self.create_config_tab()
        self.create_plan_tab()
        self.create_result_tab()

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 连接信号
        self.connect_signals()

        # 加载默认配置
        self.load_default_config()

        # 初始更新图层列表
        self.update_layer_list()

        # 失败数据
        self.failure_data = []

    def create_layer_tab(self):
        """创建图层选择选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 网络类型选择
        net_type_group = QGroupBox("当前配置网络")
        net_type_layout = QHBoxLayout(net_type_group)
        self.radio_net_4g = QRadioButton("4G")
        self.radio_net_5g = QRadioButton("5G")
        self.radio_net_4g.setChecked(True)
        net_type_layout.addWidget(self.radio_net_4g)
        net_type_layout.addWidget(self.radio_net_5g)
        net_type_layout.addStretch()
        layout.addWidget(net_type_group)

        # 图层选择
        layer_group = QGroupBox("图层选择")
        layer_layout = QVBoxLayout(layer_group)

        self.lbl_current_layer = QLabel("选择图层:")
        layer_layout.addWidget(self.lbl_current_layer)

        self.cmb_layers = QComboBox()
        self.cmb_layers.setMinimumWidth(300)
        layer_layout.addWidget(self.cmb_layers)

        self.lbl_layer_info = QLabel("未选择图层")
        self.lbl_layer_info.setStyleSheet("color: #666; font-style: italic;")
        layer_layout.addWidget(self.lbl_layer_info)

        layout.addWidget(layer_group)

        # 字段映射
        mapping_group = QGroupBox("字段映射")
        mapping_layout = QVBoxLayout(mapping_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.field_combos = {}
        for field, config in STANDARD_FIELDS.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label_text = f"{field}*" if config["required"] else field
            label = QLabel(label_text)
            label.setFixedWidth(80)
            if config["required"]:
                label.setStyleSheet("color: #d32f2f;")

            combo = QComboBox()
            combo.setMinimumWidth(180)
            self.field_combos[field] = combo

            row_layout.addWidget(label)
            row_layout.addWidget(combo)
            scroll_layout.addWidget(row_widget)

        scroll.setWidget(scroll_widget)
        mapping_layout.addWidget(scroll)
        layout.addWidget(mapping_group)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.btn_refresh_layers = QPushButton("刷新图层")
        self.btn_auto_map = QPushButton("自动映射")
        self.btn_save_mapping = QPushButton("保存配置")
        btn_layout.addWidget(self.btn_refresh_layers)
        btn_layout.addWidget(self.btn_auto_map)
        btn_layout.addWidget(self.btn_save_mapping)
        layout.addLayout(btn_layout)

        # 配置状态
        self.lbl_4g_status = QLabel("4G: 未配置")
        self.lbl_5g_status = QLabel("5G: 未配置")
        self.lbl_4g_status.setStyleSheet("color: #f57c00;")
        self.lbl_5g_status.setStyleSheet("color: #f57c00;")

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.lbl_4g_status)
        status_layout.addWidget(self.lbl_5g_status)
        layout.addLayout(status_layout)

        layout.addStretch()
        self.tab_widget.addTab(widget, "🗺️ 图层配置")

    def create_config_tab(self):
        """创建配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 规划模式
        mode_group = QGroupBox("规划模式选择")
        mode_layout = QVBoxLayout(mode_group)
        self.radio_5g4g = QRadioButton("5G→4G邻区规划")
        self.radio_5gonly = QRadioButton("5G→5G邻区规划")
        self.radio_4gonly = QRadioButton("4G→4G邻区规划")
        self.radio_5g4g.setChecked(True)
        mode_layout.addWidget(self.radio_5g4g)
        mode_layout.addWidget(self.radio_5gonly)
        mode_layout.addWidget(self.radio_4gonly)
        layout.addWidget(mode_group)

        # 距离和数量限制
        dist_group = QGroupBox("距离和数量限制")
        dist_layout = QGridLayout(dist_group)

        dist_layout.addWidget(QLabel("宏站最大距离(米):"), 0, 0)
        self.le_macro_dist = QLineEdit("2000.0")
        dist_layout.addWidget(self.le_macro_dist, 0, 1)

        dist_layout.addWidget(QLabel("宏站最大数量:"), 0, 2)
        self.le_macro_neighbors = QLineEdit("64")
        dist_layout.addWidget(self.le_macro_neighbors, 0, 3)

        dist_layout.addWidget(QLabel("室分最大距离(米):"), 1, 0)
        self.le_indoor_dist = QLineEdit("800.0")
        dist_layout.addWidget(self.le_indoor_dist, 1, 1)

        dist_layout.addWidget(QLabel("室分最大数量:"), 1, 2)
        self.le_indoor_neighbors = QLineEdit("36")
        dist_layout.addWidget(self.le_indoor_neighbors, 1, 3)

        layout.addWidget(dist_group)

        # 方位角匹配
        azimuth_group = QGroupBox("方位角匹配配置")
        azimuth_layout = QVBoxLayout(azimuth_group)
        self.cb_azimuth_match = QCheckBox("启用方位角匹配")
        azimuth_layout.addWidget(self.cb_azimuth_match)

        weight_group = QGroupBox("权重配置")
        weight_layout = QGridLayout(weight_group)
        weight_layout.addWidget(QLabel("距离权重:"), 0, 0)
        self.le_distance_weight = QLineEdit("1.0")
        self.le_distance_weight.setEnabled(False)
        weight_layout.addWidget(self.le_distance_weight, 0, 1)

        weight_layout.addWidget(QLabel("覆盖权重:"), 1, 0)
        self.le_coverage_weight = QLineEdit("0.0")
        self.le_coverage_weight.setEnabled(False)
        weight_layout.addWidget(self.le_coverage_weight, 1, 1)
        azimuth_layout.addWidget(weight_group)
        self.weight_group = weight_group
        self.weight_group.setVisible(False)

        coverage_group = QGroupBox("覆盖参数配置")
        coverage_layout = QGridLayout(coverage_group)

        coverage_layout.addWidget(QLabel("宏站覆盖半径(米):"), 0, 0)
        self.le_macro_range = QLineEdit("1500.0")
        self.le_macro_range.setEnabled(False)
        coverage_layout.addWidget(self.le_macro_range, 0, 1)

        coverage_layout.addWidget(QLabel("宏站波瓣宽度(度):"), 0, 2)
        self.le_macro_width = QLineEdit("65.0")
        self.le_macro_width.setEnabled(False)
        coverage_layout.addWidget(self.le_macro_width, 0, 3)

        coverage_layout.addWidget(QLabel("室分覆盖半径(米):"), 1, 0)
        self.le_indoor_range = QLineEdit("300.0")
        self.le_indoor_range.setEnabled(False)
        coverage_layout.addWidget(self.le_indoor_range, 1, 1)

        coverage_layout.addWidget(QLabel("室分波瓣宽度(度):"), 1, 2)
        self.le_indoor_width = QLineEdit("90.0")
        self.le_indoor_width.setEnabled(False)
        coverage_layout.addWidget(self.le_indoor_width, 1, 3)

        azimuth_layout.addWidget(coverage_group)
        self.coverage_group = coverage_group
        self.coverage_group.setVisible(False)

        layout.addWidget(azimuth_group)

        # 配置按钮
        btn_layout = QHBoxLayout()
        self.btn_save_config = QPushButton("保存配置")
        self.btn_reset_config = QPushButton("恢复默认")
        btn_layout.addWidget(self.btn_save_config)
        btn_layout.addWidget(self.btn_reset_config)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.tab_widget.addTab(widget, "⚙️ 参数配置")

    def create_plan_tab(self):
        """创建规划选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("目标小区输入")
        input_layout = QVBoxLayout(input_group)
        self.txt_target_cells = QTextEdit()
        self.txt_target_cells.setPlaceholderText("每行输入一个小区，格式：小区名称 或 基站ID-小区ID")
        input_layout.addWidget(self.txt_target_cells)
        layout.addWidget(input_group)

        input_btn_layout = QHBoxLayout()
        self.btn_paste = QPushButton("粘贴")
        self.btn_clear_input = QPushButton("清空")
        self.btn_validate = QPushButton("验证")
        input_btn_layout.addWidget(self.btn_paste)
        input_btn_layout.addWidget(self.btn_clear_input)
        input_btn_layout.addWidget(self.btn_validate)
        layout.addLayout(input_btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.lbl_plan_status = QLabel("就绪")
        layout.addWidget(self.lbl_plan_status)

        self.btn_start_plan = QPushButton("开始规划")
        self.btn_start_plan.setStyleSheet("font-size: 14px; padding: 8px; background-color: #4CAF50; color: white;")
        layout.addWidget(self.btn_start_plan)

        # 新增：当前配置预览区域
        config_preview_group = QGroupBox("当前配置预览")
        config_preview_layout = QVBoxLayout(config_preview_group)

        # 规划模式预览
        self.lbl_mode_preview = QLabel("规划模式: 未设置")
        config_preview_layout.addWidget(self.lbl_mode_preview)

        # 匹配模式预览
        self.lbl_match_preview = QLabel("匹配模式: 未设置")
        config_preview_layout.addWidget(self.lbl_match_preview)

        # 距离限制预览
        self.lbl_dist_preview = QLabel("距离限制: 未设置")
        config_preview_layout.addWidget(self.lbl_dist_preview)

        # 数量限制预览
        self.lbl_count_preview = QLabel("数量限制: 未设置")
        config_preview_layout.addWidget(self.lbl_count_preview)

        layout.addWidget(config_preview_group)

        layout.addStretch()
        self.tab_widget.addTab(widget, "📊 邻区规划")

        # 初始更新配置预览
        self.update_config_preview()

    def create_result_tab(self):
        """创建结果选项卡 - 优化按钮布局"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 结果表格
        self.table_result = QTableWidget()
        self.table_result.setColumnCount(len(EXPORT_COLUMNS))
        self.table_result.setHorizontalHeaderLabels(EXPORT_COLUMNS)
        self.table_result.horizontalHeader().setStretchLastSection(True)
        self.table_result.setAlternatingRowColors(True)
        layout.addWidget(self.table_result)

        # 按钮组
        btn_layout = QHBoxLayout()

        # 左侧：结果操作按钮
        self.btn_export_csv = QPushButton("📤 导出CSV")
        self.btn_export_csv.setToolTip("导出规划结果为CSV文件")
        self.btn_copy_row = QPushButton("📋 复制选中行")
        self.btn_copy_row.setToolTip("复制当前选中的行到剪贴板")
        self.btn_clear_result = QPushButton("🗑️ 清空结果")
        self.btn_clear_result.setToolTip("清空所有规划结果")

        # 右侧：失败报表按钮
        self.btn_show_failures = QPushButton("📝 查看失败清单")
        self.btn_show_failures.setToolTip("查看规划失败的小区清单")
        self.btn_export_failures = QPushButton("📊 导出失败清单")
        self.btn_export_failures.setToolTip("导出失败清单为CSV文件")

        # 设置按钮样式
        self.btn_export_csv.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)

        self.btn_copy_row.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)

        self.btn_clear_result.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-weight: bold;
                color: #d32f2f;
            }
            QPushButton:hover {
                background-color: #ffcdd2;
            }
        """)

        self.btn_show_failures.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-weight: bold;
                background-color: #FF9800;
                color: white;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
            QPushButton:hover:enabled {
                background-color: #F57C00;
            }
        """)

        self.btn_export_failures.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-weight: bold;
                background-color: #F44336;
                color: white;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
            QPushButton:hover:enabled {
                background-color: #D32F2F;
            }
        """)

        # 初始禁用失败清单按钮
        self.btn_show_failures.setEnabled(False)
        self.btn_export_failures.setEnabled(False)

        # 添加按钮到布局
        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_copy_row)
        btn_layout.addWidget(self.btn_clear_result)
        btn_layout.addStretch()  # 添加弹簧，将两组按钮分开
        btn_layout.addWidget(self.btn_show_failures)
        btn_layout.addWidget(self.btn_export_failures)

        layout.addLayout(btn_layout)

        # 统计信息
        self.lbl_result_stats = QLabel("暂无结果")
        self.lbl_result_stats.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #f5f5f5;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.lbl_result_stats)

        self.tab_widget.addTab(widget, "📈 结果展示")

    def connect_signals(self):
        """连接所有信号"""
        # 网络类型切换
        self.radio_net_4g.toggled.connect(self.switch_current_net_type)
        self.radio_net_5g.toggled.connect(self.switch_current_net_type)

        # 图层选择
        self.btn_refresh_layers.clicked.connect(self.update_layer_list)
        self.cmb_layers.currentIndexChanged.connect(self.on_layer_changed)

        # 字段映射按钮
        self.btn_auto_map.clicked.connect(self.auto_map_fields)
        self.btn_save_mapping.clicked.connect(self.save_current_mapping)

        # 规划模式
        self.radio_5g4g.toggled.connect(self.switch_analysis_mode)
        self.radio_5gonly.toggled.connect(self.switch_analysis_mode)
        self.radio_4gonly.toggled.connect(self.switch_analysis_mode)

        # 配置参数
        self.cb_azimuth_match.stateChanged.connect(self.update_match_mode)
        self.btn_save_config.clicked.connect(self.save_config)
        self.btn_reset_config.clicked.connect(self.load_default_config)

        # 规划操作
        self.btn_paste.clicked.connect(self.paste_from_clipboard)
        self.btn_clear_input.clicked.connect(lambda: self.txt_target_cells.clear())
        self.btn_validate.clicked.connect(self.validate_target_format)
        self.btn_start_plan.clicked.connect(self.start_planning)

        # 结果操作
        self.btn_export_csv.clicked.connect(lambda: self.export_result("csv"))
        self.btn_copy_row.clicked.connect(self.copy_selected_row)
        self.btn_clear_result.clicked.connect(self.clear_results)

        # 失败清单操作
        self.btn_show_failures.clicked.connect(self.show_failure_report)
        self.btn_export_failures.clicked.connect(self.export_failure_csv)

        # 配置变更时更新预览
        self.radio_5g4g.toggled.connect(self.update_config_preview)
        self.radio_5gonly.toggled.connect(self.update_config_preview)
        self.radio_4gonly.toggled.connect(self.update_config_preview)
        self.cb_azimuth_match.stateChanged.connect(self.update_config_preview)
        self.le_macro_dist.textChanged.connect(self.update_config_preview)
        self.le_indoor_dist.textChanged.connect(self.update_config_preview)
        self.le_macro_neighbors.textChanged.connect(self.update_config_preview)
        self.le_indoor_neighbors.textChanged.connect(self.update_config_preview)

    def switch_current_net_type(self):
        """切换当前配置的网络类型"""
        if self.radio_net_4g.isChecked():
            self.plugin.current_net_type = "4g"
            self.lbl_current_layer.setText("选择4G图层:")
        else:
            self.plugin.current_net_type = "5g"
            self.lbl_current_layer.setText("选择5G图层:")

        # 更新图层下拉框
        self.update_layer_list()

        # 恢复已选择的图层
        selected_layer = self.plugin.selected_4g_layer if self.plugin.current_net_type == "4g" else self.plugin.selected_5g_layer
        if selected_layer:
            index = self.cmb_layers.findText(selected_layer.name())
            if index >= 0:
                self.cmb_layers.setCurrentIndex(index)

    def update_layer_list(self):
        """更新图层列表"""
        # 清空现有列表
        self.cmb_layers.clear()
        self.cmb_layers.addItem("(选择图层)")

        # 获取QGIS项目中的所有点图层
        project = QgsProject.instance()
        layers = project.mapLayers().values()

        point_layers = []
        for layer in layers:
            if layer.type() == QgsVectorLayer.VectorLayer and layer.geometryType() == QgsWkbTypes.PointGeometry:
                point_layers.append(layer)

        # 添加到下拉框
        for layer in point_layers:
            self.cmb_layers.addItem(layer.name(), layer)

        # 更新字段下拉框
        if point_layers:
            self.update_field_combos()

        self.status_bar.showMessage(f"发现 {len(point_layers)} 个点图层")

    def on_layer_changed(self, index):
        """图层选择改变"""
        if index > 0:
            layer = self.cmb_layers.currentData()
            net_type = self.plugin.current_net_type

            # 保存选择的图层
            if net_type == "4g":
                self.plugin.selected_4g_layer = layer
            else:
                self.plugin.selected_5g_layer = layer

            # 更新图层信息
            feature_count = layer.featureCount()
            field_count = len(layer.fields())
            self.lbl_layer_info.setText(f"要素数: {feature_count} | 字段数: {field_count}")

            # 更新字段下拉框
            self.update_field_combos()

            # 更新状态显示
            self.update_status_display()

            self.status_bar.showMessage(f"已选择{net_type.upper()}图层: {layer.name()}")

    def update_field_combos(self):
        """更新字段下拉框"""
        # 获取当前选择的图层
        net_type = self.plugin.current_net_type
        layer = self.plugin.selected_4g_layer if net_type == "4g" else self.plugin.selected_5g_layer

        if not layer:
            for combo in self.field_combos.values():
                combo.clear()
                combo.addItem("(未选择)")
            return

        # 获取图层字段
        fields = layer.fields()
        field_names = [field.name() for field in fields]

        # 更新字段下拉框
        for field, combo in self.field_combos.items():
            combo.clear()
            combo.addItem("(未选择)")
            combo.addItems(field_names)

            # 恢复已保存的映射
            saved_mapping = self.plugin.saved_mapping[net_type]
            if field in saved_mapping and saved_mapping[field] in field_names:
                index = combo.findText(saved_mapping[field])
                if index >= 0:
                    combo.setCurrentIndex(index)

    def update_status_display(self):
        """更新状态显示"""
        # 4G状态
        if self.plugin.selected_4g_layer:
            feature_count = self.plugin.selected_4g_layer.featureCount()
            self.lbl_4g_status.setText(f"4G: 已配置 ({feature_count}个小区)")
            self.lbl_4g_status.setStyleSheet("color: #388e3c;")
        else:
            self.lbl_4g_status.setText("4G: 未配置")
            self.lbl_4g_status.setStyleSheet("color: #f57c00;")

        # 5G状态
        if self.plugin.selected_5g_layer:
            feature_count = self.plugin.selected_5g_layer.featureCount()
            self.lbl_5g_status.setText(f"5G: 已配置 ({feature_count}个小区)")
            self.lbl_5g_status.setStyleSheet("color: #388e3c;")
        else:
            self.lbl_5g_status.setText("5G: 未配置")
            self.lbl_5g_status.setStyleSheet("color: #f57c00;")

    def auto_map_fields(self):
        """自动映射字段"""
        # 获取当前选择的图层
        net_type = self.plugin.current_net_type
        layer = self.plugin.selected_4g_layer if net_type == "4g" else self.plugin.selected_5g_layer

        if not layer:
            QMessageBox.warning(self, "警告", "请先选择图层！")
            return

        # 获取图层字段
        fields = layer.fields()
        field_names = [field.name() for field in fields]

        # 使用正则表达式匹配
        for std_field, config in STANDARD_FIELDS.items():
            combo = self.field_combos.get(std_field)
            if not combo:
                continue

            # 尝试找到匹配的字段
            matched_field = None
            for pattern in config["patterns"]:
                regex = re.compile(pattern, re.IGNORECASE)
                for field_name in field_names:
                    if regex.search(field_name):
                        matched_field = field_name
                        break
                if matched_field:
                    break

            # 设置下拉框选择
            if matched_field:
                index = combo.findText(matched_field)
                if index >= 0:
                    combo.setCurrentIndex(index)

        self.status_bar.showMessage("字段自动映射完成")

    def save_current_mapping(self):
        """保存当前字段映射"""
        net_type = self.plugin.current_net_type

        # 检查图层选择
        layer = self.plugin.selected_4g_layer if net_type == "4g" else self.plugin.selected_5g_layer
        if not layer:
            QMessageBox.warning(self, "警告", f"请先选择{net_type.upper()}图层！")
            return

        # 保存字段映射
        for field, combo in self.field_combos.items():
            selected_text = combo.currentText()
            if selected_text != "(未选择)":
                self.plugin.saved_mapping[net_type][field] = selected_text
            else:
                self.plugin.saved_mapping[net_type][field] = ""

        # 检查必填字段
        missing = []
        for field, config in STANDARD_FIELDS.items():
            if config["required"] and not self.plugin.saved_mapping[net_type][field]:
                missing.append(field)

        if missing:
            QMessageBox.warning(self, "警告", f"缺失必填字段：{', '.join(missing)}")
            return

        # 构建缓存
        self.build_cell_cache(layer, net_type)

        # 更新状态显示
        self.update_status_display()

        self.status_bar.showMessage(f"{net_type.upper()}字段映射已保存")
        QMessageBox.information(self, "成功", f"{net_type.upper()}字段映射已保存！")

    def build_cell_cache(self, layer, net_type):
        """构建小区缓存"""
        cache = {"features": {}, "cell_names": set(), "spatial_index": QgsSpatialIndex(), "feature_id_to_name": {}}
        mapping = self.plugin.saved_mapping[net_type]

        # 获取字段索引
        cell_name_idx = layer.fields().indexFromName(mapping["小区名称"])
        lat_idx = layer.fields().indexFromName(mapping["纬度"])
        lon_idx = layer.fields().indexFromName(mapping["经度"])
        station_id_idx = layer.fields().indexFromName(mapping["基站ID"]) if mapping["基站ID"] else -1
        cell_id_idx = layer.fields().indexFromName(mapping["小区ID"]) if mapping["小区ID"] else -1
        type_idx = layer.fields().indexFromName(mapping["覆盖类型"]) if mapping["覆盖类型"] else -1
        freq_idx = layer.fields().indexFromName(mapping["频点"]) if mapping["频点"] else -1
        azimuth_idx = layer.fields().indexFromName(mapping["方位角"]) if mapping["方位角"] else -1
        subnet_idx = layer.fields().indexFromName(mapping["子网ID"]) if mapping.get("子网ID") else -1
        ne_idx = layer.fields().indexFromName(mapping["网元ID"]) if mapping.get("网元ID") else -1

        if cell_name_idx == -1 or lat_idx == -1 or lon_idx == -1:
            QMessageBox.warning(self, "警告", f"{net_type.upper()}图层缺少必要字段！")
            return

        try:
            spatial_index = cache["spatial_index"]
            feature_id_to_name = cache["feature_id_to_name"]
            features = cache["features"]

            # 遍历图层中的所有要素
            for feature in layer.getFeatures():
                # 获取小区名称
                cell_name = normalize_string(feature[cell_name_idx])
                if not cell_name:
                    continue

                # 获取坐标
                try:
                    lat = float(feature[lat_idx])
                    lon = float(feature[lon_idx])
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        continue
                except:
                    continue

                point = QgsPointXY(lon, lat)

                # 获取其他信息
                cell_type = "宏站"
                if type_idx != -1 and feature[type_idx]:
                    cell_type = normalize_string(feature[type_idx])

                station_id = ""
                if station_id_idx != -1 and feature[station_id_idx]:
                    station_id = normalize_string(feature[station_id_idx])

                cell_id = ""
                if cell_id_idx != -1 and feature[cell_id_idx]:
                    cell_id = normalize_string(feature[cell_id_idx])

                frequency = ""
                if freq_idx != -1 and feature[freq_idx]:
                    frequency = normalize_string(feature[freq_idx])

                azimuth = 0.0
                if azimuth_idx != -1 and feature[azimuth_idx]:
                    try:
                        azimuth = float(feature[azimuth_idx])
                    except:
                        azimuth = 0.0

                subnet_id = ""
                if subnet_idx != -1 and feature[subnet_idx]:
                    subnet_id = normalize_string(feature[subnet_idx])

                ne_id = ""
                if ne_idx != -1 and feature[ne_idx]:
                    ne_id = normalize_string(feature[ne_idx])

                # 保存到缓存
                features[cell_name] = {
                    "lat": lat,
                    "lon": lon,
                    "point": point,
                    "type": cell_type,
                    "station_id": station_id,
                    "cell_id": cell_id,
                    "cell_name": cell_name,
                    "frequency": frequency,
                    "azimuth": azimuth,
                    "subnet_id": subnet_id,
                    "ne_id": ne_id
                }
                cache["cell_names"].add(cell_name)

                # 添加到空间索引
                temp_geom = QgsGeometry.fromPointXY(point)
                spatial_index.addFeature(feature)
                feature_id_to_name[feature.id()] = cell_name

            # 更新缓存
            self.plugin.cell_cache[net_type] = cache
            self.status_bar.showMessage(f"{net_type.upper()}缓存构建完成：{len(features)} 个小区，空间索引已创建")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"构建{net_type.upper()}缓存失败：{str(e)}")

    def switch_analysis_mode(self):
        """切换分析模式"""
        if self.radio_5g4g.isChecked():
            self.plugin.analysis_mode = "5g_4g"
        elif self.radio_5gonly.isChecked():
            self.plugin.analysis_mode = "5g_only"
        else:
            self.plugin.analysis_mode = "4g_only"

        self.update_config_preview()

    def update_match_mode(self):
        """更新匹配模式 - 设置归一化权重"""
        self.plugin.config["use_azimuth_match"] = self.cb_azimuth_match.isChecked()

        if self.plugin.config["use_azimuth_match"]:
            self.weight_group.setVisible(True)
            self.coverage_group.setVisible(True)
            self.le_distance_weight.setEnabled(True)
            self.le_coverage_weight.setEnabled(True)
            self.le_macro_range.setEnabled(True)
            self.le_macro_width.setEnabled(True)
            self.le_indoor_range.setEnabled(True)
            self.le_indoor_width.setEnabled(True)

            # 设置归一化权重
            self.le_distance_weight.setText("0.5")
            self.le_coverage_weight.setText("0.5")

            # 确保权重之和为1
            self.plugin.config["distance_weight"] = 0.5
            self.plugin.config["coverage_weight"] = 0.5
        else:
            self.weight_group.setVisible(False)
            self.coverage_group.setVisible(False)
            # 禁用方位角模式时，只使用距离权重
            self.plugin.config["distance_weight"] = 1.0
            self.plugin.config["coverage_weight"] = 0.0

        self.update_config_preview()

    def update_config_preview(self):
        """更新配置预览"""
        # 规划模式
        if self.radio_5g4g.isChecked():
            mode_text = "5G→4G邻区规划"
        elif self.radio_5gonly.isChecked():
            mode_text = "5G→5G邻区规划"
        else:
            mode_text = "4G→4G邻区规划"

        self.lbl_mode_preview.setText(f"规划模式: {mode_text}")

        # 匹配模式
        match_mode = "方位角匹配模式" if self.cb_azimuth_match.isChecked() else "纯距离匹配模式"
        self.lbl_match_preview.setText(f"匹配模式: {match_mode}")

        # 距离限制
        try:
            macro_dist = float(self.le_macro_dist.text())
            indoor_dist = float(self.le_indoor_dist.text())
            self.lbl_dist_preview.setText(f"距离限制: 宏站={macro_dist}m, 室分={indoor_dist}m")
        except:
            self.lbl_dist_preview.setText("距离限制: 未设置")

        # 数量限制
        try:
            macro_count = int(self.le_macro_neighbors.text())
            indoor_count = int(self.le_indoor_neighbors.text())
            self.lbl_count_preview.setText(f"数量限制: 宏站={macro_count}个, 室分={indoor_count}个")
        except:
            self.lbl_count_preview.setText("数量限制: 未设置")

    def load_default_config(self):
        """加载默认配置"""
        self.plugin.config = {
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

        # 更新UI
        self.le_macro_dist.setText("2000.0")
        self.le_indoor_dist.setText("800.0")
        self.le_macro_neighbors.setText("64")
        self.le_indoor_neighbors.setText("36")
        self.cb_azimuth_match.setChecked(False)
        self.le_distance_weight.setText("1.0")
        self.le_coverage_weight.setText("0.0")
        self.le_macro_range.setText("1500.0")
        self.le_macro_width.setText("65.0")
        self.le_indoor_range.setText("300.0")
        self.le_indoor_width.setText("90.0")

        self.update_match_mode()
        self.update_config_preview()
        self.status_bar.showMessage("已恢复默认配置")

    def save_config(self):
        """保存配置 - 添加权重归一化处理"""
        try:
            self.plugin.config["macro_max_dist"] = float(self.le_macro_dist.text())
            self.plugin.config["indoor_max_dist"] = float(self.le_indoor_dist.text())
            self.plugin.config["macro_max_neighbors"] = int(self.le_macro_neighbors.text())
            self.plugin.config["indoor_max_neighbors"] = int(self.le_indoor_neighbors.text())

            if self.plugin.config["use_azimuth_match"]:
                # 读取权重值
                distance_weight = float(self.le_distance_weight.text())
                coverage_weight = float(self.le_coverage_weight.text())

                # 归一化处理：确保权重之和为1
                total_weight = distance_weight + coverage_weight

                if total_weight <= 0:
                    # 如果总和为0或负数，使用默认权重
                    distance_weight = 0.5
                    coverage_weight = 0.5
                    total_weight = 1.0
                    QMessageBox.warning(self, "警告", "权重值无效，已恢复默认权重(0.5, 0.5)")
                elif abs(total_weight - 1.0) > 0.001:
                    # 归一化处理
                    distance_weight = distance_weight / total_weight
                    coverage_weight = coverage_weight / total_weight
                    QMessageBox.information(self, "提示",
                                            f"权重已自动归一化：距离权重={distance_weight:.2f}，覆盖权重={coverage_weight:.2f}")

                # 更新UI显示归一化后的权重
                self.le_distance_weight.setText(f"{distance_weight:.2f}")
                self.le_coverage_weight.setText(f"{coverage_weight:.2f}")

                # 保存归一化后的权重
                self.plugin.config["distance_weight"] = distance_weight
                self.plugin.config["coverage_weight"] = coverage_weight

                # 保存其他参数
                self.plugin.config["macro_coverage_range"] = float(self.le_macro_range.text())
                self.plugin.config["macro_lobe_width"] = float(self.le_macro_width.text())
                self.plugin.config["indoor_coverage_range"] = float(self.le_indoor_range.text())
                self.plugin.config["indoor_lobe_width"] = float(self.le_indoor_width.text())

            self.update_config_preview()
            self.status_bar.showMessage("配置已保存")
            QMessageBox.information(self, "成功", "配置参数已保存！")

        except ValueError as e:
            QMessageBox.warning(self, "警告", f"参数格式错误：{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败：{str(e)}")

    def validate_target_format(self):
        """验证目标小区格式"""
        input_text = self.txt_target_cells.toPlainText()
        if not input_text:
            QMessageBox.warning(self, "警告", "请输入目标小区！")
            return

        cells = [line.strip() for line in input_text.split('\n') if line.strip()]
        if not cells:
            QMessageBox.warning(self, "警告", "请输入有效小区！")
            return

        # 简单验证
        valid_count = len([c for c in cells if c])
        QMessageBox.information(self, "验证结果", f"共 {len(cells)} 个小区，有效 {valid_count} 个")

    def paste_from_clipboard(self):
        """从剪贴板粘贴"""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            self.txt_target_cells.setText(text)
            self.status_bar.showMessage("已从剪贴板粘贴内容")
        else:
            QMessageBox.warning(self, "警告", "剪贴板为空！")

    def start_planning(self):
        """开始规划 - 增加检测上次规划结果机制"""
        # 检查输入
        input_text = self.txt_target_cells.toPlainText()
        if not input_text:
            QMessageBox.warning(self, "警告", "请输入目标小区！")
            return

        self.plugin.target_cells = [line.strip() for line in input_text.split('\n') if line.strip()]
        if not self.plugin.target_cells:
            QMessageBox.warning(self, "警告", "请输入有效小区！")
            return

        # 检查数据缓存
        source_net = "5g" if self.plugin.analysis_mode in ["5g_4g", "5g_only"] else "4g"
        target_net = "4g" if self.plugin.analysis_mode == "5g_4g" else source_net

        if not self.plugin.cell_cache[source_net]["features"]:
            QMessageBox.warning(self, "警告", f"请先配置{source_net.upper()}图层字段映射并保存！")
            return

        if not self.plugin.cell_cache[target_net]["features"]:
            QMessageBox.warning(self, "警告", f"请先配置{target_net.upper()}图层字段映射并保存！")
            return

        # 检查是否存在上次规划结果
        if self.table_result.rowCount() > 0 or len(self.plugin.plan_results) > 0:
            reply = QMessageBox.question(
                self, "发现上次规划结果",
                f"检测到有 {self.table_result.rowCount()} 条规划结果记录。\n"
                "是否要清空上次结果重新规划？\n\n"
                "• 点击【是】清空结果并开始新规划\n"
                "• 点击【否】保留结果并追加新规划\n"
                "• 点击【取消】返回",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Cancel:
                self.status_bar.showMessage("用户取消规划操作")
                return
            elif reply == QMessageBox.Yes:
                # 清空结果
                self.table_result.setRowCount(0)
                self.plugin.plan_results = []
                self.lbl_result_stats.setText("暂无结果")
                self.failure_data = []
                self.btn_show_failures.setEnabled(False)
                self.btn_export_failures.setEnabled(False)
                self.status_bar.showMessage("已清空上次规划结果")
            # 如果选择"No"，则保留结果，新结果将追加到后面

        # 清空失败数据（无论是否清空结果，失败数据都应该重新开始）
        self.failure_data = []

        # 禁用失败清单按钮
        self.btn_show_failures.setEnabled(False)
        self.btn_export_failures.setEnabled(False)

        # 禁用开始按钮，显示进度
        self.btn_start_plan.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_plan_status.setText("规划中...")

        # 创建并启动规划线程
        self.planning_thread = PlanningThread(
            self.plugin.target_cells, self.plugin.analysis_mode,
            self.plugin.config, self.plugin.cell_cache,
            self.plugin.saved_mapping
        )

        # 连接信号
        self.planning_thread.progress_update.connect(self.update_planning_progress)
        self.planning_thread.result_update.connect(self.add_planning_result)
        self.planning_thread.finish_signal.connect(self.finish_planning)
        self.planning_thread.error_signal.connect(self.handle_planning_error)
        self.planning_thread.failure_report.connect(self.handle_failure_report)

        self.planning_thread.start()

    def update_planning_progress(self, progress, msg):
        """更新规划进度"""
        self.progress_bar.setValue(progress)
        self.lbl_plan_status.setText(msg)

    def add_planning_result(self, results):
        """添加规划结果"""
        self.plugin.plan_results.extend(results)

        # 更新表格
        current_row = self.table_result.rowCount()
        self.table_result.setRowCount(current_row + len(results))

        for row_idx, result in enumerate(results):
            for col_idx, col_name in enumerate(EXPORT_COLUMNS):
                value = result.get(col_name, "")
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table_result.setItem(current_row + row_idx, col_idx, item)

    def handle_failure_report(self, failure_list):
        """处理失败报告"""
        self.failure_data = failure_list

    def finish_planning(self, stats, plan_time):
        """完成规划 - 优化交互体验"""
        self.btn_start_plan.setEnabled(True)
        self.progress_bar.setValue(100)

        stats_text = f"""
        处理小区：{stats['total_processed']}个 | 成功：{stats['success_count']}个 | 失败：{stats['failed_count']}个
        生成邻区：{stats['total_neighbors']}个 | 平均得分：{stats['avg_score']:.4f}
        规划耗时：{plan_time:.2f}秒
        """.strip()

        # 如果有失败数据，更新统计信息并启用失败清单按钮
        if self.failure_data:
            stats_text += f"\n失败小区数：{len(self.failure_data)}个（已记录失败清单）"
            self.btn_show_failures.setEnabled(True)
            self.btn_export_failures.setEnabled(True)
        else:
            self.btn_show_failures.setEnabled(False)
            self.btn_export_failures.setEnabled(False)

        self.lbl_plan_status.setText("✅ 规划完成！")
        self.lbl_result_stats.setText(stats_text)
        self.status_bar.showMessage(f"✅ 规划完成，耗时 {plan_time:.2f} 秒")

        # 总是显示规划完成提示框，无论成功还是失败
        if self.failure_data:
            # 有失败的提示框
            reply = QMessageBox.information(
                self, "规划完成",
                f"✅ 规划已完成！\n\n"
                f"📊 统计信息：\n"
                f"• 处理小区：{stats['total_processed']}个\n"
                f"• 成功规划：{stats['success_count']}个\n"
                f"• 失败：{stats['failed_count']}个\n"
                f"• 生成邻区：{stats['total_neighbors']}个\n"
                f"• 平均得分：{stats['avg_score']:.4f}\n"
                f"• 耗时：{plan_time:.2f}秒\n\n"
                f"⚠️ 有{len(self.failure_data)}个小区规划失败。",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Ignore,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # 切换到结果选项卡
                self.tab_widget.setCurrentIndex(3)  # 结果展示选项卡
                # 自动显示失败清单
                self.show_failure_report()
            elif reply == QMessageBox.No:
                # 只切换到结果选项卡
                self.tab_widget.setCurrentIndex(3)  # 结果展示选项卡
        else:
            # 完全成功的提示框
            reply = QMessageBox.information(
                self, "规划完成",
                f"✅ 规划已完成！\n\n"
                f"📊 统计信息：\n"
                f"• 处理小区：{stats['total_processed']}个\n"
                f"• 成功规划：{stats['success_count']}个\n"
                f"• 失败：{stats['failed_count']}个\n"
                f"• 生成邻区：{stats['total_neighbors']}个\n"
                f"• 平均得分：{stats['avg_score']:.4f}\n"
                f"• 耗时：{plan_time:.2f}秒\n\n"
                f"🎉 所有小区规划成功！",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # 切换到结果选项卡
                self.tab_widget.setCurrentIndex(3)  # 结果展示选项卡

    def handle_planning_error(self, error_msg):
        """处理规划错误"""
        self.btn_start_plan.setEnabled(True)
        self.lbl_plan_status.setText("❌ 规划失败！")
        self.status_bar.showMessage(f"❌ 规划失败：{error_msg}")
        QMessageBox.critical(self, "错误", f"❌ 规划失败：{error_msg}")

    def export_result(self, export_type):
        """导出结果 - 仅支持CSV格式"""
        if self.table_result.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有数据可导出！")
            return

        # 仅支持CSV格式
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件",
            f"邻区规划_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                # 写入表头
                f.write(','.join(EXPORT_COLUMNS) + '\n')

                # 写入数据
                for row in range(self.table_result.rowCount()):
                    row_data = []
                    for col in range(self.table_result.columnCount()):
                        item = self.table_result.item(row, col)
                        row_data.append(f'"{item.text()}"' if item else '')
                    f.write(','.join(row_data) + '\n')

            QMessageBox.information(self, "成功", f"✅ CSV文件已保存：\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"❌ 导出CSV失败：{str(e)}")

    def show_failure_report(self):
        """显示失败报告对话框"""
        if not self.failure_data:
            QMessageBox.information(self, "失败清单", "本次规划没有失败的小区。")
            return

        # 创建对话框显示失败清单
        dialog = QDialog(self)
        dialog.setWindowTitle(f"规划失败清单 ({len(self.failure_data)}个)")
        dialog.setMinimumSize(700, 400)

        layout = QVBoxLayout(dialog)

        # 表格显示
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["小区", "失败原因", "失败时间"])
        table.setRowCount(len(self.failure_data))

        for row, failure in enumerate(self.failure_data):
            table.setItem(row, 0, QTableWidgetItem(failure["小区"]))
            table.setItem(row, 1, QTableWidgetItem(failure["失败原因"]))
            table.setItem(row, 2, QTableWidgetItem(failure["失败时间"]))

        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        layout.addWidget(table)

        # 按钮组
        button_layout = QHBoxLayout()
        btn_export = QPushButton("📤 导出CSV")
        btn_copy = QPushButton("📋 复制到剪贴板")
        btn_close = QPushButton("关闭")

        btn_export.clicked.connect(lambda: self.export_failure_csv(dialog))
        btn_copy.clicked.connect(lambda: self.copy_failure_to_clipboard())
        btn_close.clicked.connect(dialog.close)

        button_layout.addWidget(btn_export)
        button_layout.addWidget(btn_copy)
        button_layout.addWidget(btn_close)
        layout.addLayout(button_layout)

        dialog.exec_()

    def export_failure_csv(self, parent_dialog=None):
        """导出失败清单为CSV"""
        if not self.failure_data:
            QMessageBox.warning(self, "警告", "没有失败数据可导出！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存失败清单",
            f"规划失败清单_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                # 写入表头
                f.write("小区,失败原因,失败时间\n")

                # 写入数据
                for failure in self.failure_data:
                    # 转义特殊字符
                    cell = failure["小区"].replace('"', '""')
                    reason = failure["失败原因"].replace('"', '""')
                    time_str = failure["失败时间"]

                    f.write(f'"{cell}","{reason}","{time_str}"\n')

            if parent_dialog:
                parent_dialog.close()

            QMessageBox.information(self, "成功", f"✅ 失败清单已保存：\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"❌ 导出失败清单失败：{str(e)}")

    def copy_failure_to_clipboard(self):
        """复制失败清单到剪贴板"""
        if not self.failure_data:
            QMessageBox.warning(self, "警告", "没有失败数据！")
            return

        clipboard = QApplication.clipboard()
        text = "小区\t失败原因\t失败时间\n"

        for failure in self.failure_data:
            cell = failure["小区"]
            reason = failure["失败原因"]
            time_str = failure["失败时间"]
            text += f"{cell}\t{reason}\t{time_str}\n"

        clipboard.setText(text)
        self.status_bar.showMessage("✅ 失败清单已复制到剪贴板")

    def copy_selected_row(self):
        """复制选中行"""
        selected = self.table_result.selectedItems()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要复制的行！")
            return

        row = selected[0].row()
        row_data = []
        for col in range(self.table_result.columnCount()):
            item = self.table_result.item(row, col)
            row_data.append(item.text() if item else "")

        clipboard = QApplication.clipboard()
        clipboard.setText('\t'.join(row_data))
        self.status_bar.showMessage("✅ 已复制选中行到剪贴板")

    def clear_results(self):
        """清除结果"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有规划结果吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.table_result.setRowCount(0)
            self.plugin.plan_results = []
            self.lbl_result_stats.setText("暂无结果")
            self.failure_data = []
            self.btn_show_failures.setEnabled(False)
            self.btn_export_failures.setEnabled(False)
            self.status_bar.showMessage("✅ 已清除所有结果")


# ==================== 主插件类 ====================

class CellNeighborPlannerPlugin:
    """QGIS邻区规划插件主类"""

    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.toolbar = None
        self.menu = None

        # 初始化变量
        self.target_cells = []
        self.plan_results = []

        # 小区缓存
        self.cell_cache = {
            "4g": {"features": {}, "cell_names": set(), "spatial_index": None, "feature_id_to_name": {}},
            "5g": {"features": {}, "cell_names": set(), "spatial_index": None, "feature_id_to_name": {}}
        }

        # 当前设置
        self.current_net_type = "4g"
        self.analysis_mode = "5g_4g"

        # 选择的图层
        self.selected_4g_layer = None
        self.selected_5g_layer = None

        # 字段映射
        self.saved_mapping = {
            "4g": {field: "" for field in STANDARD_FIELDS.keys()},
            "5g": {field: "" for field in STANDARD_FIELDS.keys()}
        }
        self.current_mapping = {field: "" for field in STANDARD_FIELDS.keys()}

        # 配置参数
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
        """初始化GUI"""
        # 创建菜单项
        self.menu = QMenu("邻区规划工具")
        self.iface.pluginMenu().addMenu(self.menu)

        # 创建工具栏按钮
        self.toolbar = self.iface.addToolBar("邻区规划")
        self.toolbar.setObjectName("NeighborPlanningToolbar")

        # ==== 添加这部分代码 ====
        # 获取插件目录并加载图标
        import os
        plugin_dir = os.path.dirname(__file__)
        icon_path = os.path.join(plugin_dir, 'icon.png')

        # 检查图标文件是否存在
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            # 如果图标文件不存在，使用一个简单的替代图标
            icon = QIcon()
            QgsMessageLog.logMessage(f"图标文件未找到: {icon_path}", "邻区规划工具")

        # 创建主按钮并设置图标
        self.action = QAction(icon, "邻区规划")
        # ========================

        self.action.triggered.connect(self.show_main_window)
        self.toolbar.addAction(self.action)
        self.menu.addAction(self.action)

        # 创建主窗口（但不立即显示）
        self.main_window = None

    def unload(self):
        """卸载插件"""
        # 移除菜单
        if self.menu:
            self.iface.pluginMenu().removeAction(self.menu.menuAction())
            self.menu.deleteLater()
            self.menu = None

        # 移除工具栏
        if self.toolbar:
            self.toolbar.deleteLater()
            self.toolbar = None

        # 关闭主窗口
        if self.main_window:
            self.main_window.close()
            self.main_window.deleteLater()
            self.main_window = None

    def show_main_window(self):
        """显示主窗口"""
        if not self.main_window:
            self.main_window = MainWindow(self.iface, self)

        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()