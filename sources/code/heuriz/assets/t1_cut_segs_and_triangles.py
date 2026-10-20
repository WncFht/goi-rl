import numpy as np
import re
import scipy.io  # 用于保存 .mat 文件
from earcut.earcut import earcut
import csv


def parse_polygons_from_file(path,startswith="世界"):
    polys = []
    cur = []
    pat = re.compile(startswith+r'\(\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*\)')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('--------------------------------'):
                if cur:
                    polys.append(cur)
                    cur = []
                continue
            m = pat.search(line)
            if m:
                x = float(m.group(1))
                y = float(m.group(2))
                cur.append((x, y))
    if cur:
        polys.append(cur)
    return polys



def point_in_poly(pt, poly):
    # ray casting algorithm (returns True if inside or on edge)
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        # check if point is exactly on a vertex or segment (allow small eps)
        # segment check:
        if min(y0, y1) - 1e-12 <= y <= max(y0, y1) + 1e-12 and min(x0, x1) - 1e-12 <= x <= max(x0, x1) + 1e-12:
            # collinearity test
            dx1 = x1 - x0
            dy1 = y1 - y0
            dxp = x - x0
            dyp = y - y0
            cross = dx1 * dyp - dy1 * dxp
            if abs(cross) < 1e-10:
                return True
        if ((y0 > y) != (y1 > y)):
            xinters = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xinters == x:
                return True
            if xinters > x:
                inside = not inside
    return inside


def remove_contained(polys):
    n = len(polys)
    keep = [True] * n
    for i in range(n):
        if not keep[i]: continue
        for j in range(n):
            if i == j or not keep[j]: continue
            # quick area check
            # compute bounding box of j and test if poly i bbox inside j bbox; else skip containment test
            minxi = min(p[0] for p in polys[i])
            maxxi = max(p[0] for p in polys[i])
            minyi = min(p[1] for p in polys[i])
            maxyi = max(p[1] for p in polys[i])
            minxj = min(p[0] for p in polys[j])
            maxxj = max(p[0] for p in polys[j])
            minyj = min(p[1] for p in polys[j])
            maxyj = max(p[1] for p in polys[j])
            if not (
                    minxj - 1e-12 <= minxi and maxxi <= maxxj + 1e-12 and minyj - 1e-12 <= minyi and maxyi <= maxyj + 1e-12):
                continue
            # test all vertices of i inside j
            all_in = True
            for v in polys[i]:
                if not point_in_poly(v, polys[j]):
                    all_in = False
                    break
            if all_in:
                keep[i] = False
                break
    return [polys[i] for i in range(n) if keep[i]]


def ear_clip(poly):
    data = []
    for pt in poly:
        data.extend(pt)

    indices = earcut(data, dim=2)
    tris = []
    for i in range(0, len(indices), 3):
        idx1 = indices[i]
        idx2 = indices[i + 1]
        idx3 = indices[i + 2]
        tris.append([poly[idx1][0], poly[idx1][1], poly[idx2][0], poly[idx2][1], poly[idx3][0], poly[idx3][1]])

    return tris


if __name__ == '__main__':

    postfix="_props"

    input_path =f"vertices{postfix}.txt"

    polys = parse_polygons_from_file(input_path,startswith="本地")
    print(f"解析到 {len(polys)} 多边形（含可能被包含的）。")

    polys = remove_contained(polys)
    print(f"去除被完全包含的多边形后剩余 {len(polys)} 个。")

    poly_sides = [[*p[i], *p[(i + 1) % len(p)]] for p in polys for i in range(len(p))]
    print(f"共 {len(poly_sides)} 条边。")

    with open(f'all_lines{postfix}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(poly_sides)

    all_tris = [tri for p in polys for tri in ear_clip(p)]
    print(f"耳切：生成 {len(all_tris)} 个三角形。")

    with open(f'all_triangles{postfix}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_tris)

    # 生成200*200的预制网格：每个网格需要渲染哪些三角形下标都计算好，保存为pt文件；每个网格需要考虑哪些激光雷达线段，也计算好保存为pt文件
    # 我只能手动限定好每个cell最多能渲染多少三角形、最多能考虑多少激光雷达线段，这样数据量就是死的而不是活的了
