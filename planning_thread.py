# planning_thread.py
import time
import math
import numpy as np
from datetime import datetime

from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.core import QgsPointXY

from .constants import EXPORT_COLUMNS
from .utils import normalize_angle_diff, normalize_string
from .spatial_calculator import QgisSpatialCalculator


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

        total = len(target_cells)
        self.progress_step = 5 if total < 100 else 10 if total < 1000 else 50
        self.last_progress = 0
        self.result_buffer = []
        self.result_step = 20

        self.spatial_calc = QgisSpatialCalculator()
        self.failure_list = []
        self._build_parse_index()

        # 预计算配置常量，避免重复查字典
        self._precompute_config()

    def _precompute_config(self):
        """将配置值转为本地属性，加速访问"""
        cfg = self.config
        self.macro_max_dist = cfg["macro_max_dist"]
        self.indoor_max_dist = cfg["indoor_max_dist"]
        self.macro_max_neighbors = cfg["macro_max_neighbors"]
        self.indoor_max_neighbors = cfg["indoor_max_neighbors"]
        self.use_azimuth = cfg["use_azimuth_match"]
        if self.use_azimuth:
            self.distance_weight = cfg["distance_weight"]
            self.coverage_weight = cfg["coverage_weight"]
            self.macro_cov_range = cfg["macro_coverage_range"]
            self.macro_lobe_width = cfg["macro_lobe_width"]
            self.indoor_cov_range = cfg["indoor_coverage_range"]
            self.indoor_lobe_width = cfg["indoor_lobe_width"]

    def _build_parse_index(self):
        self._parse_index = {}
        for net_type in ["4g", "5g"]:
            self._parse_index[net_type] = {}
            cache = self.cell_cache[net_type]
            for cell_name, feat in cache["features"].items():
                sid = feat.get("station_id", "")
                cid = feat.get("cell_id", "")
                if sid and cid:
                    self._parse_index[net_type][f"{sid}-{cid}"] = cell_name

    def run(self):
        try:
            all_results = []
            total = len(self.target_cells)
            stats = {"total_processed": 0, "success_count": 0, "failed_count": 0,
                     "total_neighbors": 0, "avg_score": 0.0, "plan_time": 0.0}
            start_time = time.time()

            for idx, cell_input in enumerate(self.target_cells):
                current_idx = idx + 1
                progress = int(current_idx / total * 100)
                if (current_idx % self.progress_step == 0 or current_idx == total or
                        progress - self.last_progress >= 1):
                    self.progress_update.emit(progress, f"处理中: {cell_input} ({current_idx}/{total})")
                    self.last_progress = progress

                results, failure_reason = self._plan_single_cell_optimized(cell_input, self.mode)
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
                    if failure_reason:
                        self.failure_list.append({
                            "小区": cell_input,
                            "失败原因": failure_reason,
                            "失败时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            self.progress_update.emit(100, f"处理完成：共处理{total}个小区")
            stats["plan_time"] = time.time() - start_time
            if all_results:
                scores = [r["综合得分"] for r in all_results if isinstance(r.get("综合得分"), (int, float))]
                if scores:
                    stats["avg_score"] = np.mean(scores)
            if self.failure_list:
                self.failure_report.emit(self.failure_list)
            self.finish_signal.emit(stats, stats["plan_time"])
        except Exception as e:
            self.error_signal.emit(str(e))

    # ----------------------------------------------------------------------
    def _plan_single_cell_optimized(self, target_input, mode):
        """核心优化：空间索引粗筛后，向量化计算所有候选的距离/方位/得分"""
        if mode == "5g_4g":
            src_net, tgt_net = "5g", "4g"
        elif mode == "5g_only":
            src_net = tgt_net = "5g"
        elif mode == "4g_only":
            src_net = tgt_net = "4g"
        else:
            return [], "未知规划模式"

        matched_cell = self.parse_target_cell(target_input, src_net)
        if not matched_cell:
            return [], f"源小区'{target_input}'在{src_net.upper()}图层中不存在"

        src_cache = self.cell_cache[src_net]
        if matched_cell not in src_cache["features"]:
            return [], f"源小区'{matched_cell}'在{src_net.upper()}缓存中不存在"

        src_feat = src_cache["features"][matched_cell]
        src_lat, src_lon = src_feat["lat"], src_feat["lon"]
        src_type = src_feat["type"]
        src_azimuth = src_feat["azimuth"]
        src_station = src_feat["station_id"]

        # 距离/数量限制
        if src_type == "宏站":
            max_dist = self.macro_max_dist
            max_neighbors = self.macro_max_neighbors
            cov_range = self.macro_cov_range if self.use_azimuth else None
            lobe_width = self.macro_lobe_width if self.use_azimuth else None
        else:
            max_dist = self.indoor_max_dist
            max_neighbors = self.indoor_max_neighbors
            cov_range = self.indoor_cov_range if self.use_azimuth else None
            lobe_width = self.indoor_lobe_width if self.use_azimuth else None

        tgt_cache = self.cell_cache[tgt_net]

        # ---- 1) 空间索引粗筛候选小区名称 ----
        rect = self.spatial_calc.create_buffered_rectangle(QgsPointXY(src_lon, src_lat), max_dist)
        candidate_names = []
        spatial_index = tgt_cache.get("spatial_index")
        if spatial_index:
            for fid in spatial_index.intersects(rect):
                name = tgt_cache["feature_id_to_name"].get(fid)
                if name and not (name == matched_cell and src_net == tgt_net):
                    candidate_names.append(name)
        else:
            for name in tgt_cache["features"]:
                if name != matched_cell or src_net != tgt_net:
                    candidate_names.append(name)

        if not candidate_names:
            return [], f"指定距离内({max_dist}米)无可用邻区"

        # ---- 2) 提取候选小区坐标数组 ----
        n_cand = len(candidate_names)
        tgt_lats = np.empty(n_cand, dtype=np.float64)
        tgt_lons = np.empty(n_cand, dtype=np.float64)
        tgt_azimuths = np.empty(n_cand, dtype=np.float64) if self.use_azimuth else None
        tgt_types = [] if self.use_azimuth else None
        tgt_stations = [""] * n_cand

        for i, name in enumerate(candidate_names):
            feat = tgt_cache["features"][name]
            tgt_lats[i] = feat["lat"]
            tgt_lons[i] = feat["lon"]
            if self.use_azimuth:
                tgt_azimuths[i] = feat["azimuth"]
                tgt_types.append(feat["type"])
            tgt_stations[i] = feat.get("station_id", "")

        # ---- 3) 向量化距离计算并筛选 ----
        distances = QgisSpatialCalculator.batch_haversine(src_lon, src_lat, tgt_lons, tgt_lats)
        dist_mask = distances <= max_dist
        if not np.any(dist_mask):
            return [], f"指定距离内({max_dist}米)无可用邻区"

        # 根据距离掩码过滤数组
        valid_idx = np.where(dist_mask)[0]
        distances = distances[valid_idx]
        filtered_names = [candidate_names[i] for i in valid_idx]
        tgt_lats_filtered = tgt_lats[valid_idx]
        tgt_lons_filtered = tgt_lons[valid_idx]
        n_valid = len(distances)

        # 准备得分数组
        if self.use_azimuth:
            tgt_azimuths = tgt_azimuths[valid_idx]
            tgt_types_valid = [tgt_types[i] for i in valid_idx]
            # 向量化方位角（源->目标 和 目标->源）
            bearings_st = QgisSpatialCalculator.batch_bearing(src_lon, src_lat, tgt_lons_filtered, tgt_lats_filtered)
            bearings_ts = QgisSpatialCalculator.batch_bearing(tgt_lons_filtered, tgt_lats_filtered, src_lon, src_lat)

            # 角度差计算
            diff_st = np.abs(bearings_st - src_azimuth)
            diff_st = np.minimum(diff_st, 360 - diff_st)  # 0-180
            diff_ts = np.abs(bearings_ts - tgt_azimuths)
            diff_ts = np.minimum(diff_ts, 360 - diff_ts)

            # 覆盖得分批量计算
            coverage_scores = self._batch_coverage_score(distances, diff_st, diff_ts,
                                                         src_type, tgt_types_valid,
                                                         cov_range, lobe_width)
            # 方位角匹配得分
            azimuth_scores = self._batch_azimuth_match(diff_st, diff_ts, src_azimuth, tgt_azimuths)
            # 距离得分（批量）
            distance_scores = self._batch_distance_score(distances, src_type, tgt_types_valid)

            # 综合得分 = 距离*权重 + 覆盖*权重 + 方位角匹配*权重
            total_scores = (distance_scores * self.distance_weight +
                            coverage_scores * self.coverage_weight +
                            azimuth_scores * self.coverage_weight)
        else:
            # 纯距离模式
            distance_scores = self._batch_distance_score(distances, src_type, [None]*n_valid)  # target_type不重要
            total_scores = distance_scores
            coverage_scores = np.zeros(n_valid)
            azimuth_scores = np.zeros(n_valid)

        # 同站加分（向量化）
        same_site_bonus = np.zeros(n_valid)
        if src_station:
            for i in range(n_valid):
                name = filtered_names[i]
                if tgt_stations[valid_idx[i]] == src_station and distances[i] <= 100:
                    same_site_bonus[i] = 0.2
        total_scores = np.clip(total_scores + same_site_bonus, 0.0, 1.0)

        # ---- 4) 选出 Top-N 并构建结果 ----
        # 获取排序索引
        sort_idx = np.argsort(-total_scores)  # 降序
        top_k = min(max_neighbors, n_valid)
        selected_indices = sort_idx[:top_k]

        final_results = []
        for idx in selected_indices:
            name = filtered_names[idx]
            tgt_feat = tgt_cache["features"][name]
            # 构造候选信息
            cand = {
                "cell_name": name,
                "feature": tgt_feat,
                "distance": distances[idx],
                "total_score": total_scores[idx],
                "components": {
                    "distance_score": distance_scores[idx],
                    "coverage_score": coverage_scores[idx],
                    "azimuth_match_score": azimuth_scores[idx],
                    "same_site_bonus": same_site_bonus[idx]
                },
                "neighbor_type": self._determine_neighbor_type(distances[idx], total_scores[idx], self.use_azimuth)
            }
            final_results.append(self.build_result_data(src_feat, tgt_feat, src_net, tgt_net, cand))

        return final_results, None

    # ----------------------------------------------------------------------
    # 批量得分函数（NumPy 实现）
    def _batch_distance_score(self, distances, src_type, tgt_types):
        """向量化距离得分。tgt_types 可为 None 或列表"""
        if src_type == "宏站":
            max_d = self.macro_max_dist
        else:
            max_d = self.indoor_max_dist
        # 简单处理：所有距离用相同 max_d（宏站之间可能不同，但原始代码也用源类型决定 max_d）
        # 如果需要精确宏站-室分混合，可进一步优化，此处保持与原始逻辑一致（源类型决定）
        scores = np.where(distances <= 50, 1.0,
                          np.where(distances <= max_d,
                                   np.exp(-0.7 * (distances - 50) / (max_d - 50)),
                                   0.0))
        return scores

    def _batch_coverage_score(self, distances, diff_st, diff_ts, src_type, tgt_types, cov_range, lobe_width):
        """
        diff_st, diff_ts: 角度差数组 (0-180)
        返回覆盖得分数组
        """
        # 距离因子
        d_factor = np.where(distances <= cov_range,
                            np.exp(-(distances / cov_range) ** 2),
                            np.maximum(0, 1 - (distances - cov_range) / cov_range))
        # 角度因子
        half = lobe_width / 2
        # 源侧角度因子
        a_st = np.where(diff_st <= half,
                        1.0 - (diff_st / half) * 0.4,
                        np.where(diff_st <= half * 2,
                                 0.6 - ((diff_st - half) / half) * 0.4,
                                 np.maximum(0.0, 0.2 - ((diff_st - half * 2) / 180) * 0.2)))
        # 目标侧角度因子
        a_ts = np.where(diff_ts <= half,
                        1.0 - (diff_ts / half) * 0.4,
                        np.where(diff_ts <= half * 2,
                                 0.6 - ((diff_ts - half) / half) * 0.4,
                                 np.maximum(0.0, 0.2 - ((diff_ts - half * 2) / 180) * 0.2)))
        cov = np.sqrt(d_factor * a_st * d_factor * a_ts)  # 实际上是 sqrt(d_factor^2 * a_st * a_ts) = d_factor * sqrt(a_st * a_ts)
        # 修正：原始代码 coverage_score = sqrt(coverage_st * coverage_ts) 其中 coverage_st = distance_factor * angle_factor_st
        # 所以 = sqrt(d_factor * a_st * d_factor * a_ts) = d_factor * sqrt(a_st * a_ts)
        scores = d_factor * np.sqrt(a_st * a_ts)
        return np.clip(scores, 0.0, 1.0)

    def _batch_azimuth_match(self, diff_st, diff_ts, src_az, tgt_azs):
        """向量化方位角匹配得分"""
        def single_score(ang):
            return np.where(ang <= 30, 1.0 - (ang / 30) * 0.3,
                            np.where(ang <= 60, 0.7 - ((ang - 30) / 30) * 0.3,
                                     np.where(ang <= 90, 0.4 - ((ang - 60) / 30) * 0.2,
                                              np.where(ang <= 120, 0.2 - ((ang - 90) / 30) * 0.1,
                                                       np.maximum(0.0, 0.1 - ((ang - 120) / 60) * 0.1)))))
        # 处理零方位角
        zero_mask = (src_az == 0.0) | (tgt_azs == 0.0)
        score_st = single_score(diff_st)
        score_ts = single_score(diff_ts)
        match = np.sqrt(score_st * score_ts)
        match[zero_mask] = 0.5
        return match

    def _determine_neighbor_type(self, dist, score, use_azimuth):
        if dist <= 500:
            base = "近距离"
        elif dist <= 1500:
            base = "中距离"
        else:
            base = "远距离"
        if score >= 0.8:
            rel = "强相关"
        elif score >= 0.6:
            rel = "中等相关"
        elif score >= 0.4:
            rel = "弱相关"
        else:
            rel = "极弱相关"
        tag = "方位角匹配" if use_azimuth else "纯距离"
        return f"{base}-{rel}-{tag}"

    # build_result_data 与原版完全一致，此处省略（请从上一版复制）
    def build_result_data(self, src, tgt, src_net, tgt_net, cand):
        return {
            "源网络类型": "5G" if src_net == "5g" else "4G",
            "目标网络类型": "5G" if tgt_net == "5g" else "4G",
            "源子网ID": src.get("subnet_id", ""),
            "源网元ID": src.get("ne_id", ""),
            "源基站ID": src.get("station_id", ""),
            "源小区ID": src.get("cell_id", ""),
            "源小区名称": src.get("cell_name", ""),
            "源覆盖类型": src.get("type", ""),
            "源频点": src.get("frequency", ""),
            "源方位角": src.get("azimuth", ""),
            "目标子网ID": tgt.get("subnet_id", ""),
            "目标网元ID": tgt.get("ne_id", ""),
            "目标基站ID": tgt.get("station_id", ""),
            "目标小区ID": tgt.get("cell_id", ""),
            "目标小区名称": tgt.get("cell_name", ""),
            "目标覆盖类型": tgt.get("type", ""),
            "目标频点": tgt.get("frequency", ""),
            "目标方位角": tgt.get("azimuth", ""),
            "距离(m)": round(cand["distance"], 2),
            "覆盖相关度": round(cand["components"]["coverage_score"], 4),
            "方位角匹配度": round(cand["components"]["azimuth_match_score"], 4),
            "综合得分": round(cand["total_score"], 4),
            "邻区类型": cand["neighbor_type"],
            "规划时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def parse_target_cell(self, cell_input, net_type):
        cell_input = cell_input.strip()
        if not cell_input:
            return None
        cache = self.cell_cache[net_type]
        if cell_input in cache["features"]:
            return cell_input
        if '-' in cell_input:
            hit = self._parse_index.get(net_type, {}).get(cell_input)
            if hit:
                return hit
        for name in cache["features"]:
            if cell_input in name:
                return name
        return None