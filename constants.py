# constants.py

# 标准字段定义：键为内部标准名称，值为配置（是否必填，匹配正则列表）
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

# 导出结果列顺序
EXPORT_COLUMNS = [
    "源网络类型", "目标网络类型", "源子网ID", "源网元ID", "源基站ID", "源小区ID", "源小区名称",
    "源覆盖类型", "源频点", "源方位角", "目标子网ID", "目标网元ID", "目标基站ID", "目标小区ID",
    "目标小区名称", "目标覆盖类型", "目标频点", "目标方位角", "距离(m)",
    "覆盖相关度", "方位角匹配度", "综合得分", "邻区类型", "规划时间"
]