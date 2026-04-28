# utils.py
import math

def normalize_angle_diff(angle1: float, angle2: float) -> float:
    """计算两个角度之间的最小角度差（0-180度）"""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff

def normalize_string(s):
    """去除首尾空白，空值返回空字符串"""
    if s is None:
        return ""
    return str(s).strip()