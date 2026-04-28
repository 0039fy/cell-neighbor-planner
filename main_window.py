# main_window.py
import re
from datetime import datetime
import numpy as np

from qgis.PyQt.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QGroupBox, QLabel, QComboBox, QLineEdit, QTextEdit,
    QPushButton, QRadioButton, QCheckBox, QProgressBar, QStatusBar,
    QScrollArea, QTableWidget, QTableWidgetItem, QMessageBox,
    QFileDialog, QDialog, QApplication
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsWkbTypes, QgsSpatialIndex,
    QgsGeometry, QgsFeature, QgsPointXY
)

from .constants import STANDARD_FIELDS, EXPORT_COLUMNS
from .planning_thread import PlanningThread
from .utils import normalize_string

# ====== 兼容 QGIS 3.x 和 4.x 的枚举常量 ======
# PyQt5: Qt.AlignCenter 有效；PyQt6: 必须用 Qt.AlignmentFlag.AlignCenter
if hasattr(Qt, 'AlignCenter'):
    ALIGN_CENTER = Qt.AlignCenter
else:
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter

# QMessageBox 按钮兼容（已在全部位置使用 StandardButton，此处保留以防万一）
# 但为了代码统一，下面全部用 QMessageBox.StandardButton

# QgsWkbTypes.PointGeometry 在 QGIS 4 中可能迁移至 Qgis.WkbType，但当前错误未涉及，
# 若后续报错，可用类似方式兼容：
# if hasattr(QgsWkbTypes, 'PointGeometry'):
#     POINT_GEOM = QgsWkbTypes.PointGeometry
# else:
#     from qgis.core import Qgis
#     POINT_GEOM = Qgis.WkbType.PointGeometry


class MainWindow(QMainWindow):
    def __init__(self, iface, plugin):
        super().__init__()
        self.iface = iface
        self.plugin = plugin

        self.setWindowTitle("邻区规划工具 v3.1 - 优化版")
        self.setMinimumSize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.create_layer_tab()
        self.create_config_tab()
        self.create_plan_tab()
        self.create_result_tab()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.connect_signals()
        self.load_default_config()
        self.update_layer_list()

        self.failure_data = []

    # ---------- 图层选项卡 ----------
    def create_layer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        net_group = QGroupBox("当前配置网络")
        net_layout = QHBoxLayout(net_group)
        self.radio_net_4g = QRadioButton("4G")
        self.radio_net_5g = QRadioButton("5G")
        self.radio_net_4g.setChecked(True)
        net_layout.addWidget(self.radio_net_4g)
        net_layout.addWidget(self.radio_net_5g)
        net_layout.addStretch()
        layout.addWidget(net_group)

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

        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新图层")
        self.btn_auto_map = QPushButton("自动映射")
        self.btn_save_mapping = QPushButton("保存配置")
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_auto_map)
        btn_layout.addWidget(self.btn_save_mapping)
        layout.addLayout(btn_layout)

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

    # ---------- 配置选项卡 ----------
    def create_config_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

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

        azimuth_group = QGroupBox("方位角匹配配置")
        azimuth_layout = QVBoxLayout(azimuth_group)
        self.cb_azimuth_match = QCheckBox("启用方位角匹配")
        azimuth_layout.addWidget(self.cb_azimuth_match)

        self.weight_group = QGroupBox("权重配置")
        weight_layout = QGridLayout(self.weight_group)
        weight_layout.addWidget(QLabel("距离权重:"), 0, 0)
        self.le_distance_weight = QLineEdit("1.0")
        self.le_distance_weight.setEnabled(False)
        weight_layout.addWidget(self.le_distance_weight, 0, 1)
        weight_layout.addWidget(QLabel("覆盖权重:"), 1, 0)
        self.le_coverage_weight = QLineEdit("0.0")
        self.le_coverage_weight.setEnabled(False)
        weight_layout.addWidget(self.le_coverage_weight, 1, 1)
        azimuth_layout.addWidget(self.weight_group)
        self.weight_group.setVisible(False)

        self.coverage_group = QGroupBox("覆盖参数配置")
        coverage_layout = QGridLayout(self.coverage_group)
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
        azimuth_layout.addWidget(self.coverage_group)
        self.coverage_group.setVisible(False)

        layout.addWidget(azimuth_group)

        btn_layout = QHBoxLayout()
        self.btn_save_config = QPushButton("保存配置")
        self.btn_reset_config = QPushButton("恢复默认")
        btn_layout.addWidget(self.btn_save_config)
        btn_layout.addWidget(self.btn_reset_config)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.tab_widget.addTab(widget, "⚙️ 参数配置")

    # ---------- 规划选项卡 ----------
    def create_plan_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("目标小区输入")
        input_layout = QVBoxLayout(input_group)
        self.txt_target_cells = QTextEdit()
        self.txt_target_cells.setPlaceholderText("每行一个小区，格式：小区名称 或 基站ID-小区ID")
        input_layout.addWidget(self.txt_target_cells)
        layout.addWidget(input_group)

        btn_layout = QHBoxLayout()
        self.btn_paste = QPushButton("粘贴")
        self.btn_clear_input = QPushButton("清空")
        self.btn_validate = QPushButton("验证")
        btn_layout.addWidget(self.btn_paste)
        btn_layout.addWidget(self.btn_clear_input)
        btn_layout.addWidget(self.btn_validate)
        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.lbl_plan_status = QLabel("就绪")
        layout.addWidget(self.lbl_plan_status)

        self.btn_start_plan = QPushButton("开始规划")
        self.btn_start_plan.setStyleSheet(
            "font-size: 14px; padding: 8px; background-color: #4CAF50; color: white;"
        )
        layout.addWidget(self.btn_start_plan)

        config_prev = QGroupBox("当前配置预览")
        prev_layout = QVBoxLayout(config_prev)
        self.lbl_mode_preview = QLabel("规划模式: 未设置")
        self.lbl_match_preview = QLabel("匹配模式: 未设置")
        self.lbl_dist_preview = QLabel("距离限制: 未设置")
        self.lbl_count_preview = QLabel("数量限制: 未设置")
        prev_layout.addWidget(self.lbl_mode_preview)
        prev_layout.addWidget(self.lbl_match_preview)
        prev_layout.addWidget(self.lbl_dist_preview)
        prev_layout.addWidget(self.lbl_count_preview)
        layout.addWidget(config_prev)

        layout.addStretch()
        self.tab_widget.addTab(widget, "📊 邻区规划")
        self.update_config_preview()

    # ---------- 结果选项卡 ----------
    def create_result_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.table_result = QTableWidget()
        self.table_result.setColumnCount(len(EXPORT_COLUMNS))
        self.table_result.setHorizontalHeaderLabels(EXPORT_COLUMNS)
        self.table_result.horizontalHeader().setStretchLastSection(True)
        self.table_result.setAlternatingRowColors(True)
        layout.addWidget(self.table_result)

        btn_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("📤 导出CSV")
        self.btn_export_csv.setToolTip("导出规划结果为CSV文件")
        self.btn_copy_row = QPushButton("📋 复制选中行")
        self.btn_copy_row.setToolTip("复制当前选中的行到剪贴板")
        self.btn_clear_result = QPushButton("🗑️ 清空结果")
        self.btn_clear_result.setToolTip("清空所有规划结果")

        self.btn_show_failures = QPushButton("📝 查看失败清单")
        self.btn_show_failures.setToolTip("查看规划失败的小区清单")
        self.btn_export_failures = QPushButton("📊 导出失败清单")
        self.btn_export_failures.setToolTip("导出失败清单为CSV文件")

        # 按钮样式（原版恢复）
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
            QPushButton:hover:!disabled {
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
            QPushButton:hover:!disabled {
                background-color: #D32F2F;
            }
        """)

        self.btn_show_failures.setEnabled(False)
        self.btn_export_failures.setEnabled(False)

        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_copy_row)
        btn_layout.addWidget(self.btn_clear_result)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_show_failures)
        btn_layout.addWidget(self.btn_export_failures)
        layout.addLayout(btn_layout)

        self.lbl_result_stats = QLabel("暂无结果")
        self.lbl_result_stats.setStyleSheet(
            "padding: 8px; background-color: #f5f5f5; border-radius: 4px; font-weight: bold;"
        )
        layout.addWidget(self.lbl_result_stats)

        self.tab_widget.addTab(widget, "📈 结果展示")

    # ---------- 信号连接 ----------
    def connect_signals(self):
        self.radio_net_4g.toggled.connect(self.switch_current_net_type)
        self.radio_net_5g.toggled.connect(self.switch_current_net_type)
        self.btn_refresh.clicked.connect(self.update_layer_list)
        self.cmb_layers.currentIndexChanged.connect(self.on_layer_changed)
        self.btn_auto_map.clicked.connect(self.auto_map_fields)
        self.btn_save_mapping.clicked.connect(self.save_current_mapping)

        self.radio_5g4g.toggled.connect(self.switch_analysis_mode)
        self.radio_5gonly.toggled.connect(self.switch_analysis_mode)
        self.radio_4gonly.toggled.connect(self.switch_analysis_mode)

        self.cb_azimuth_match.stateChanged.connect(self.update_match_mode)
        self.btn_save_config.clicked.connect(self.save_config)
        self.btn_reset_config.clicked.connect(self.load_default_config)

        self.btn_paste.clicked.connect(self.paste_from_clipboard)
        self.btn_clear_input.clicked.connect(lambda: self.txt_target_cells.clear())
        self.btn_validate.clicked.connect(self.validate_target_format)
        self.btn_start_plan.clicked.connect(self.start_planning)

        self.btn_export_csv.clicked.connect(self.export_result)
        self.btn_copy_row.clicked.connect(self.copy_selected_row)
        self.btn_clear_result.clicked.connect(self.clear_results)
        self.btn_show_failures.clicked.connect(self.show_failure_report)
        self.btn_export_failures.clicked.connect(self.export_failure_csv)

        # 配置预览更新
        for w in [self.radio_5g4g, self.radio_5gonly, self.radio_4gonly,
                  self.cb_azimuth_match, self.le_macro_dist, self.le_indoor_dist,
                  self.le_macro_neighbors, self.le_indoor_neighbors]:
            if isinstance(w, QRadioButton) or isinstance(w, QCheckBox):
                w.toggled.connect(self.update_config_preview)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self.update_config_preview)

    # ---------- 图层相关 ----------
    def switch_current_net_type(self):
        if self.radio_net_4g.isChecked():
            self.plugin.current_net_type = "4g"
            self.lbl_current_layer.setText("选择4G图层:")
        else:
            self.plugin.current_net_type = "5g"
            self.lbl_current_layer.setText("选择5G图层:")
        self.update_layer_list()
        selected = self.plugin.selected_4g_layer if self.plugin.current_net_type == "4g" else self.plugin.selected_5g_layer
        if selected:
            idx = self.cmb_layers.findText(selected.name())
            if idx >= 0:
                self.cmb_layers.setCurrentIndex(idx)

    def update_layer_list(self):
        self.cmb_layers.clear()
        self.cmb_layers.addItem("(选择图层)")
        project = QgsProject.instance()
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.PointGeometry:
                self.cmb_layers.addItem(layer.name(), layer)
        if self.cmb_layers.count() > 1:
            self.update_field_combos()

    def on_layer_changed(self, index):
        if index > 0:
            layer = self.cmb_layers.currentData()
            net = self.plugin.current_net_type
            if net == "4g":
                self.plugin.selected_4g_layer = layer
            else:
                self.plugin.selected_5g_layer = layer
            self.lbl_layer_info.setText(f"要素数: {layer.featureCount()} | 字段数: {len(layer.fields())}")
            self.update_field_combos()
            self.update_status_display()

    def update_field_combos(self):
        net = self.plugin.current_net_type
        layer = self.plugin.selected_4g_layer if net == "4g" else self.plugin.selected_5g_layer
        if not layer:
            for combo in self.field_combos.values():
                combo.clear()
                combo.addItem("(未选择)")
            return
        field_names = [f.name() for f in layer.fields()]
        for std_field, combo in self.field_combos.items():
            combo.clear()
            combo.addItem("(未选择)")
            combo.addItems(field_names)
            saved = self.plugin.saved_mapping[net].get(std_field, "")
            if saved in field_names:
                combo.setCurrentIndex(combo.findText(saved))

    def update_status_display(self):
        for net, lbl in [("4g", self.lbl_4g_status), ("5g", self.lbl_5g_status)]:
            layer = self.plugin.selected_4g_layer if net == "4g" else self.plugin.selected_5g_layer
            if layer:
                lbl.setText(f"{net.upper()}: 已配置 ({layer.featureCount()}个小区)")
                lbl.setStyleSheet("color: #388e3c;")
            else:
                lbl.setText(f"{net.upper()}: 未配置")
                lbl.setStyleSheet("color: #f57c00;")

    def auto_map_fields(self):
        net = self.plugin.current_net_type
        layer = self.plugin.selected_4g_layer if net == "4g" else self.plugin.selected_5g_layer
        if not layer:
            QMessageBox.warning(self, "警告", "请先选择图层！")
            return
        field_names = [f.name() for f in layer.fields()]
        for std_field, config in STANDARD_FIELDS.items():
            combo = self.field_combos.get(std_field)
            if not combo:
                continue
            matched = None
            for pattern in config["patterns"]:
                regex = re.compile(pattern, re.IGNORECASE)
                for fname in field_names:
                    if regex.search(fname):
                        matched = fname
                        break
                if matched:
                    break
            if matched:
                combo.setCurrentIndex(combo.findText(matched))
        self.status_bar.showMessage("字段自动映射完成")

    def save_current_mapping(self):
        net = self.plugin.current_net_type
        layer = self.plugin.selected_4g_layer if net == "4g" else self.plugin.selected_5g_layer
        if not layer:
            QMessageBox.warning(self, "警告", f"请先选择{net.upper()}图层！")
            return
        for field, combo in self.field_combos.items():
            val = combo.currentText()
            self.plugin.saved_mapping[net][field] = val if val != "(未选择)" else ""
        missing = [f for f, c in STANDARD_FIELDS.items() if c["required"] and not self.plugin.saved_mapping[net][f]]
        if missing:
            QMessageBox.warning(self, "警告", f"缺失必填字段：{', '.join(missing)}")
            return
        self.build_cell_cache(layer, net)
        self.update_status_display()
        self.status_bar.showMessage(f"{net.upper()}字段映射已保存")
        QMessageBox.information(self, "成功", f"{net.upper()}字段映射已保存！")

    # ---------- 缓存构建 ----------
    def build_cell_cache(self, layer, net_type):
        cache = {
            "features": {},
            "cell_names": set(),
            "spatial_index": QgsSpatialIndex(),
            "feature_id_to_name": {},
            "names": np.array([], dtype=object),
            "lats": np.array([], dtype=np.float64),
            "lons": np.array([], dtype=np.float64)
        }
        mapping = self.plugin.saved_mapping[net_type]
        fields = layer.fields()
        idx = {}
        for key in ["小区名称", "纬度", "经度", "基站ID", "小区ID", "覆盖类型", "频点", "方位角", "子网ID", "网元ID"]:
            if mapping.get(key):
                idx[key] = fields.indexFromName(mapping[key])
            else:
                idx[key] = -1

        if idx["小区名称"] == -1:
            QMessageBox.warning(self, "警告", f"{net_type.upper()}图层缺少小区名称字段！")
            return

        try:
            features_list = []
            for feat in layer.getFeatures():
                cell_name = normalize_string(feat[idx["小区名称"]])
                if not cell_name:
                    continue

                geom = feat.geometry()
                if geom and not geom.isEmpty() and geom.type() == QgsWkbTypes.PointGeometry:
                    pt = geom.asPoint()
                    lon, lat = pt.x(), pt.y()
                else:
                    if idx["纬度"] == -1 or idx["经度"] == -1:
                        continue
                    try:
                        lat = float(feat[idx["纬度"]])
                        lon = float(feat[idx["经度"]])
                    except (ValueError, TypeError):
                        continue
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                point = QgsPointXY(lon, lat)

                station_id = normalize_string(feat[idx["基站ID"]]) if idx["基站ID"] != -1 else ""
                cell_id = normalize_string(feat[idx["小区ID"]]) if idx["小区ID"] != -1 else ""
                cell_type = normalize_string(feat[idx["覆盖类型"]]) if idx["覆盖类型"] != -1 else "宏站"
                frequency = normalize_string(feat[idx["频点"]]) if idx["频点"] != -1 else ""
                try:
                    azimuth = float(feat[idx["方位角"]]) if idx["方位角"] != -1 else 0.0
                except (ValueError, TypeError):
                    azimuth = 0.0
                subnet_id = normalize_string(feat[idx["子网ID"]]) if idx["子网ID"] != -1 else ""
                ne_id = normalize_string(feat[idx["网元ID"]]) if idx["网元ID"] != -1 else ""

                feat_info = {
                    "lat": lat, "lon": lon, "point": point,
                    "type": cell_type, "station_id": station_id,
                    "cell_id": cell_id, "cell_name": cell_name,
                    "frequency": frequency, "azimuth": azimuth,
                    "subnet_id": subnet_id, "ne_id": ne_id
                }
                cache["features"][cell_name] = feat_info
                cache["cell_names"].add(cell_name)
                features_list.append((cell_name, lat, lon))

                temp_feat = QgsFeature(feat.id())
                temp_feat.setGeometry(QgsGeometry.fromPointXY(point))
                cache["spatial_index"].addFeature(temp_feat)
                cache["feature_id_to_name"][feat.id()] = cell_name

            if features_list:
                names_arr, lats_arr, lons_arr = zip(*features_list)
                cache["names"] = np.array(names_arr, dtype=object)
                cache["lats"] = np.array(lats_arr, dtype=np.float64)
                cache["lons"] = np.array(lons_arr, dtype=np.float64)
            else:
                cache["names"] = np.array([], dtype=object)
                cache["lats"] = np.array([], dtype=np.float64)
                cache["lons"] = np.array([], dtype=np.float64)

            self.plugin.cell_cache[net_type] = cache
            self.status_bar.showMessage(f"{net_type.upper()}缓存构建完成：{len(cache['features'])} 个小区")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"构建{net_type.upper()}缓存失败：{str(e)}")

    # ---------- 配置管理 ----------
    def switch_analysis_mode(self):
        if self.radio_5g4g.isChecked():
            self.plugin.analysis_mode = "5g_4g"
        elif self.radio_5gonly.isChecked():
            self.plugin.analysis_mode = "5g_only"
        else:
            self.plugin.analysis_mode = "4g_only"
        self.update_config_preview()

    def update_match_mode(self):
        use = self.cb_azimuth_match.isChecked()
        self.plugin.config["use_azimuth_match"] = use
        self.weight_group.setVisible(use)
        self.coverage_group.setVisible(use)
        for w in [self.le_distance_weight, self.le_coverage_weight,
                  self.le_macro_range, self.le_macro_width,
                  self.le_indoor_range, self.le_indoor_width]:
            w.setEnabled(use)
        if use:
            self.le_distance_weight.setText("0.5")
            self.le_coverage_weight.setText("0.5")
        else:
            self.plugin.config["distance_weight"] = 1.0
            self.plugin.config["coverage_weight"] = 0.0
        self.update_config_preview()

    def update_config_preview(self):
        if self.radio_5g4g.isChecked():
            mode_txt = "5G→4G邻区规划"
        elif self.radio_5gonly.isChecked():
            mode_txt = "5G→5G邻区规划"
        else:
            mode_txt = "4G→4G邻区规划"
        self.lbl_mode_preview.setText(f"规划模式: {mode_txt}")
        match_mode = "方位角匹配模式" if self.cb_azimuth_match.isChecked() else "纯距离匹配模式"
        self.lbl_match_preview.setText(f"匹配模式: {match_mode}")
        try:
            md = float(self.le_macro_dist.text())
            id_ = float(self.le_indoor_dist.text())
            self.lbl_dist_preview.setText(f"距离限制: 宏站={md}m, 室分={id_}m")
        except:
            self.lbl_dist_preview.setText("距离限制: 格式错误")
        try:
            mn = int(self.le_macro_neighbors.text())
            inn = int(self.le_indoor_neighbors.text())
            self.lbl_count_preview.setText(f"数量限制: 宏站={mn}个, 室分={inn}个")
        except:
            self.lbl_count_preview.setText("数量限制: 格式错误")

    def load_default_config(self):
        self.plugin.config = {
            "macro_max_dist": 2000.0, "indoor_max_dist": 800.0,
            "macro_max_neighbors": 64, "indoor_max_neighbors": 36,
            "use_azimuth_match": False, "distance_weight": 1.0,
            "coverage_weight": 0.0, "macro_coverage_range": 1500.0,
            "indoor_coverage_range": 300.0, "macro_lobe_width": 65.0,
            "indoor_lobe_width": 90.0
        }
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
        try:
            self.plugin.config["macro_max_dist"] = float(self.le_macro_dist.text())
            self.plugin.config["indoor_max_dist"] = float(self.le_indoor_dist.text())
            self.plugin.config["macro_max_neighbors"] = int(self.le_macro_neighbors.text())
            self.plugin.config["indoor_max_neighbors"] = int(self.le_indoor_neighbors.text())
            if self.plugin.config["use_azimuth_match"]:
                dw = float(self.le_distance_weight.text())
                cw = float(self.le_coverage_weight.text())
                total = dw + cw
                if total <= 0:
                    dw, cw = 0.5, 0.5
                    total = 1.0
                    QMessageBox.warning(self, "警告", "权重无效，已重置为默认(0.5,0.5)")
                if abs(total - 1.0) > 0.001:
                    dw /= total
                    cw /= total
                    QMessageBox.information(self, "提示", f"权重已归一化: 距离={dw:.2f}, 覆盖={cw:.2f}")
                self.le_distance_weight.setText(f"{dw:.2f}")
                self.le_coverage_weight.setText(f"{cw:.2f}")
                self.plugin.config["distance_weight"] = dw
                self.plugin.config["coverage_weight"] = cw
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
            QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    # ---------- 规划触发 ----------
    def validate_target_format(self):
        text = self.txt_target_cells.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "请输入目标小区！")
            return
        cells = [l.strip() for l in text.split('\n') if l.strip()]
        QMessageBox.information(self, "验证结果", f"共 {len(cells)} 个小区")

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            self.txt_target_cells.setText(text)
            self.status_bar.showMessage("已从剪贴板粘贴")
        else:
            QMessageBox.warning(self, "警告", "剪贴板为空！")

    def start_planning(self):
        input_text = self.txt_target_cells.toPlainText().strip()
        if not input_text:
            QMessageBox.warning(self, "警告", "请输入目标小区！")
            return
        cells = [l.strip() for l in input_text.split('\n') if l.strip()]
        if not cells:
            QMessageBox.warning(self, "警告", "请输入有效小区！")
            return
        self.plugin.target_cells = cells

        mode = self.plugin.analysis_mode
        if mode in ("5g_4g", "5g_only"):
            src_net = "5g"
        else:
            src_net = "4g"
        tgt_net = "4g" if mode == "5g_4g" else src_net
        if not self.plugin.cell_cache[src_net]["features"]:
            QMessageBox.warning(self, "警告", f"请先配置{src_net.upper()}图层并保存！")
            return
        if not self.plugin.cell_cache[tgt_net]["features"]:
            QMessageBox.warning(self, "警告", f"请先配置{tgt_net.upper()}图层并保存！")
            return

        if self.table_result.rowCount() > 0:
            reply = QMessageBox.question(
                self, "发现上次结果",
                f"当前有 {self.table_result.rowCount()} 条记录。是否清空并重新规划？\n是 - 清空并重新规划\n否 - 追加结果",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self.table_result.setRowCount(0)
                self.plugin.plan_results = []
                self.failure_data = []
                self.btn_show_failures.setEnabled(False)
                self.btn_export_failures.setEnabled(False)

        self.failure_data = []
        self.btn_show_failures.setEnabled(False)
        self.btn_export_failures.setEnabled(False)
        self.btn_start_plan.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_plan_status.setText("规划中...")

        self.planning_thread = PlanningThread(
            cells, mode, self.plugin.config,
            self.plugin.cell_cache, self.plugin.saved_mapping
        )
        self.planning_thread.progress_update.connect(self.update_planning_progress)
        self.planning_thread.result_update.connect(self.add_planning_result)
        self.planning_thread.finish_signal.connect(self.finish_planning)
        self.planning_thread.error_signal.connect(self.handle_planning_error)
        self.planning_thread.failure_report.connect(self.handle_failure_report)
        self.planning_thread.start()

    def update_planning_progress(self, progress, msg):
        self.progress_bar.setValue(progress)
        self.lbl_plan_status.setText(msg)

    def add_planning_result(self, results):
        self.plugin.plan_results.extend(results)
        row = self.table_result.rowCount()
        self.table_result.setRowCount(row + len(results))
        for i, res in enumerate(results):
            for col, key in enumerate(EXPORT_COLUMNS):
                item = QTableWidgetItem(str(res.get(key, "")))
                item.setTextAlignment(ALIGN_CENTER)   # 使用兼容常量
                self.table_result.setItem(row + i, col, item)

    def handle_failure_report(self, failure_list):
        self.failure_data = failure_list

    def finish_planning(self, stats, plan_time):
        self.btn_start_plan.setEnabled(True)
        self.progress_bar.setValue(100)
        stats_text = (
            f"处理小区：{stats['total_processed']}个 | 成功：{stats['success_count']}个 | "
            f"失败：{stats['failed_count']}个\n"
            f"生成邻区：{stats['total_neighbors']}个 | 平均得分：{stats['avg_score']:.4f}\n"
            f"规划耗时：{plan_time:.2f}秒"
        )
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

        msg = (f"✅ 规划完成！\n处理 {stats['total_processed']} 个小区，生成 {stats['total_neighbors']} 个邻区。"
               f"\n耗时 {plan_time:.2f} 秒。")
        if self.failure_data:
            msg += f"\n⚠️ 有 {len(self.failure_data)} 个小区失败，可查看失败清单。"
        QMessageBox.information(self, "规划完成", msg)
        self.tab_widget.setCurrentIndex(3)

    def handle_planning_error(self, error_msg):
        self.btn_start_plan.setEnabled(True)
        self.lbl_plan_status.setText("❌ 规划失败！")
        self.status_bar.showMessage(f"❌ 规划失败：{error_msg}")
        QMessageBox.critical(self, "错误", f"❌ 规划失败：{error_msg}")

    # ---------- 结果导出与操作 ----------
    def export_result(self):
        if self.table_result.rowCount() == 0:
            QMessageBox.warning(self, "警告", "没有数据可导出！")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV",
            f"邻区规划_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8-sig') as f:
                f.write(','.join(EXPORT_COLUMNS) + '\n')
                for row in range(self.table_result.rowCount()):
                    items = []
                    for col in range(self.table_result.columnCount()):
                        it = self.table_result.item(row, col)
                        items.append(f'"{it.text()}"' if it else '')
                    f.write(','.join(items) + '\n')
            QMessageBox.information(self, "成功", f"✅ 已保存：{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def copy_selected_row(self):
        sel = self.table_result.selectedItems()
        if not sel:
            QMessageBox.warning(self, "警告", "请先选择一行！")
            return
        row = sel[0].row()
        data = []
        for col in range(self.table_result.columnCount()):
            it = self.table_result.item(row, col)
            data.append(it.text() if it else "")
        QApplication.clipboard().setText('\t'.join(data))
        self.status_bar.showMessage("已复制到剪贴板")

    def clear_results(self):
        reply = QMessageBox.question(
            self, "确认", "确定清空所有规划结果？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table_result.setRowCount(0)
            self.plugin.plan_results = []
            self.failure_data = []
            self.btn_show_failures.setEnabled(False)
            self.btn_export_failures.setEnabled(False)
            self.lbl_result_stats.setText("暂无结果")
            self.status_bar.showMessage("已清空结果")

    def show_failure_report(self):
        if not self.failure_data:
            QMessageBox.information(self, "失败清单", "没有失败记录。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"规划失败清单 ({len(self.failure_data)}个)")
        dlg.setMinimumSize(700, 400)
        layout = QVBoxLayout(dlg)
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["小区", "失败原因", "失败时间"])
        table.setRowCount(len(self.failure_data))
        for i, f in enumerate(self.failure_data):
            table.setItem(i, 0, QTableWidgetItem(f["小区"]))
            table.setItem(i, 1, QTableWidgetItem(f["失败原因"]))
            table.setItem(i, 2, QTableWidgetItem(f["失败时间"]))
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("导出CSV")
        btn_copy = QPushButton("复制到剪贴板")
        btn_close = QPushButton("关闭")
        btn_export.clicked.connect(lambda: self.export_failure_csv(dlg))
        btn_copy.clicked.connect(lambda: self.copy_failure_to_clipboard())
        btn_close.clicked.connect(dlg.close)
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        dlg.exec()  # 兼容 Qt5/Qt6

    def export_failure_csv(self, parent_dialog=None):
        if not self.failure_data:
            QMessageBox.warning(self, "警告", "没有失败数据！")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存失败清单",
            f"规划失败清单_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8-sig') as f:
                f.write("小区,失败原因,失败时间\n")
                for item in self.failure_data:
                    cell = item["小区"].replace('"', '""')
                    reason = item["失败原因"].replace('"', '""')
                    time_str = item["失败时间"]
                    f.write(f'"{cell}","{reason}","{time_str}"\n')
            if parent_dialog:
                parent_dialog.close()
            QMessageBox.information(self, "成功", f"失败清单已保存：{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def copy_failure_to_clipboard(self):
        if not self.failure_data:
            return
        text = "小区\t失败原因\t失败时间\n"
        for f in self.failure_data:
            text += f"{f['小区']}\t{f['失败原因']}\t{f['失败时间']}\n"
        QApplication.clipboard().setText(text)
        self.status_bar.showMessage("失败清单已复制到剪贴板")