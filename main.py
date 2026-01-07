import cv2
import numpy as np
from collections import defaultdict, deque
import os
import glob


def is_valid_brightness(color, min_threshold=60, max_threshold=210):
    """判断颜色亮度是否在有效范围内

    Args:
        color: RGB颜色值，可以是单个颜色或颜色数组
        min_threshold: 最小亮度阈值，默认60
        max_threshold: 最大亮度阈值，默认210

    Returns:
        bool: 亮度是否在有效范围内
    """
    return min_threshold <= np.mean(color) <= max_threshold


def mark_initial_patches(
    img_rgb, patch_size, grid_size, different_group_size, different_threshold
):
    """在分组中标记不同的图像块（亮度筛选）

    Args:
        img_rgb: RGB图像
        patch_size: 图像块大小 (宽, 高)
        grid_size: 网格尺寸 (行, 列)
        different_group_size: 图像块颜色差异分组大小
        different_threshold: 图像块颜色差异阈值

    Returns:
        标记的图像块列表
    """
    height, width = img_rgb.shape[:2]
    marked_patches = []

    for group_row in range(0, grid_size[0], different_group_size):
        for group_col in range(0, grid_size[1], different_group_size):
            group_patches = []
            group_positions = []

            for i in range(different_group_size):
                for j in range(different_group_size):
                    row = group_row + i
                    col = group_col + j

                    if row >= grid_size[0] or col >= grid_size[1]:
                        continue

                    x1 = col * patch_size[0]
                    y1 = row * patch_size[1]
                    x2 = min(x1 + patch_size[0], width)
                    y2 = min(y1 + patch_size[1], height)

                    patch = img_rgb[y1:y2, x1:x2]
                    if patch.size > 0:
                        patch_mean = np.mean(patch.reshape(-1, 3), axis=0)
                        group_patches.append(patch_mean)
                        group_positions.append((row, col, x1, y1, x2, y2))

            if len(group_patches) < 2:
                continue

            bg_color = np.mean(np.array(group_patches).reshape(-1, 3), axis=0)
            bg_norm = np.linalg.norm(bg_color)

            for idx, patch_color in enumerate(group_patches):
                if not is_valid_brightness(patch_color):
                    continue
                distance = np.linalg.norm(patch_color - bg_color)
                relative_diff = distance / bg_norm if bg_norm > 0 else 0
                if relative_diff >= different_threshold:
                    marked_patch = group_positions[idx]
                    marked_patches.append(marked_patch)

    return marked_patches


def check_and_add_neighbor_patch(
    patch_size,
    grid_size,
    img_rgb,
    marked_patch_color,
    marked_positions,
    similarity_threshold,
    neighbor_row,
    neighbor_col,
    color_cache=None,
):
    """检查并添加单个邻居patch

    Args:
        patch_size: 图像块大小 (宽, 高)
        grid_size: 网格尺寸 (行, 列)
        img_rgb: RGB图像
        marked_patch_color: 标记patch的颜色
        marked_positions: 已标记的位置集合
        similarity_threshold: 相似度阈值
        neighbor_row: 邻居patch的行索引
        neighbor_col: 邻居patch的列索引

        color_cache: 颜色缓存字典，避免重复计算

    Returns:
        tuple: (是否添加, 邻居patch位置) 或 (False, None)
    """
    height, width = img_rgb.shape[:2]

    if (
        neighbor_row < 0
        or neighbor_row >= grid_size[0]
        or neighbor_col < 0
        or neighbor_col >= grid_size[1]
    ):
        return False, None

    neighbor_x1 = neighbor_col * patch_size[0]
    neighbor_y1 = neighbor_row * patch_size[1]
    neighbor_x2 = min(neighbor_x1 + patch_size[0], width)
    neighbor_y2 = min(neighbor_y1 + patch_size[1], height)

    neighbor_pos = (
        neighbor_row,
        neighbor_col,
        neighbor_x1,
        neighbor_y1,
        neighbor_x2,
        neighbor_y2,
    )

    if neighbor_pos in marked_positions:
        return False, None

    neighbor_patch = img_rgb[neighbor_y1:neighbor_y2, neighbor_x1:neighbor_x2]
    if neighbor_patch.size == 0:
        return False, None

    neighbor_key = (neighbor_row, neighbor_col)
    if color_cache is not None and neighbor_key in color_cache:
        neighbor_color = color_cache[neighbor_key]
    else:
        neighbor_color = np.mean(neighbor_patch.reshape(-1, 3), axis=0)
        if color_cache is not None:
            color_cache[neighbor_key] = neighbor_color

    if not is_valid_brightness(neighbor_color):
        return False, None

    marked_norm = np.linalg.norm(marked_patch_color)
    neighbor_norm = np.linalg.norm(neighbor_color)

    if marked_norm > 0 and neighbor_norm > 0:
        similarity = 1 - (
            np.linalg.norm(marked_patch_color - neighbor_color)
            / max(marked_norm, neighbor_norm)
        )
        if similarity > similarity_threshold:
            return True, neighbor_pos
    else:
        pass

    return False, None


def find_similar_neighbors_for_patch(
    img_rgb,
    patch_size,
    grid_size,
    marked_patch,
    marked_positions,
    similarity_threshold,
    color_cache=None,
    neighbor_check_count=[0],
):
    """查找单个patch的所有相似邻居

    Args:
        patch_size: 图像块大小 (宽, 高)
        grid_size: 网格尺寸 (行, 列)
        img_rgb: RGB图像
        marked_patch: 标记的patch (row, col, x1, y1, x2, y2)
        marked_positions: 已标记的位置集合
        similarity_threshold: 相似度阈值
        color_cache: 颜色缓存字典，避免重复计算
        neighbor_check_count: 邻居检查计数器（列表，可变）

    Returns:
        新添加的邻居patch位置列表
    """
    row, col, x1, y1, x2, y2 = marked_patch
    patch_key = (row, col)

    if color_cache is not None and patch_key in color_cache:
        marked_patch_color = color_cache[patch_key]
    else:
        marked_patch_color = np.mean(img_rgb[y1:y2, x1:x2].reshape(-1, 3), axis=0)
        if color_cache is not None:
            color_cache[patch_key] = marked_patch_color

    new_patches = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            neighbor_row = row + dr
            neighbor_col = col + dc

            neighbor_check_count[0] += 1

            should_add, neighbor_pos = check_and_add_neighbor_patch(
                patch_size,
                grid_size,
                img_rgb,
                marked_patch_color,
                marked_positions,
                similarity_threshold,
                neighbor_row,
                neighbor_col,
                color_cache,
            )

            if should_add:
                new_patches.append(neighbor_pos)
                marked_positions.add(neighbor_pos)

    return new_patches


def expand_marked_patches(
    initial_marked_patches,
    img_rgb,
    patch_size,
    grid_size,
    similarity_threshold,
    max_iterations=10,
    max_total_patches=500,
):
    """扩展标记的图像块

    Args:
        initial_marked_patches: 初始标记的图像块列表
        img_rgb: RGB图像
        patch_size: 图像块大小 (宽, 高)
        grid_size: 网格尺寸 (行, 列)
        similarity_threshold: 相似度阈值
        max_iterations: 最大迭代次数，默认10
        max_total_patches: 最大总patch数量，默认500

    Returns:
        (所有标记图像块列表, 新扩展的图像块列表)
    """
    marked_positions = {
        (m[0], m[1], m[2], m[3], m[4], m[5]) for m in initial_marked_patches
    }
    queue = deque(initial_marked_patches)
    color_cache = {}

    for patch in initial_marked_patches:
        row, col, x1, y1, x2, y2 = patch
        patch_key = (row, col)
        if patch_key not in color_cache:
            color_cache[patch_key] = np.mean(
                img_rgb[y1:y2, x1:x2].reshape(-1, 3), axis=0
            )

    all_marked_patches = list(initial_marked_patches)
    expanded_marked_patches = []

    iteration = 0

    total_checked = 0
    total_found = 0
    neighbor_check_count = [0]

    while (
        queue
        and iteration < max_iterations
        and len(all_marked_patches) < max_total_patches
    ):
        iteration += 1
        current_batch_size = len(queue)
        added_in_iteration = 0

        for i in range(current_batch_size):
            if len(all_marked_patches) >= max_total_patches:
                break

            marked_patch = queue.popleft()
            row, col = marked_patch[0], marked_patch[1]

            similar_neighbors = find_similar_neighbors_for_patch(
                img_rgb,
                patch_size,
                grid_size,
                marked_patch,
                marked_positions,
                similarity_threshold,
                color_cache,
                neighbor_check_count,
            )

            total_checked += 1
            if similar_neighbors:
                all_marked_patches.extend(similar_neighbors)
                queue.extend(similar_neighbors)
                expanded_marked_patches.extend(similar_neighbors)
                added_in_iteration += len(similar_neighbors)
                total_found += len(similar_neighbors)

    return all_marked_patches, expanded_marked_patches


def calculate_density_scores(patches, grid_size, density_group_size):
    """计算每个密度组的分数

    Args:
        patches: 标记的图像块列表
        grid_size: 网格尺寸 (行, 列)
        density_group_size: 密度分组大小

    Returns:
        dict: 密度分数字典，键为(group_row, group_col)，值为patch数量
    """
    density_scores = defaultdict(int)
    for group_row in range(0, grid_size[0], density_group_size):
        for group_col in range(0, grid_size[1], density_group_size):
            for marked in patches:
                m_row, m_col = marked[0], marked[1]
                if (
                    group_row <= m_row < group_row + density_group_size
                    and group_col <= m_col < group_col + density_group_size
                ):
                    density_scores[(group_row, group_col)] += 1
    return density_scores


def get_patches_in_group(patches, group_row, group_col, density_group_size):
    """获取指定组内的所有patches

    Args:
        patches: 标记的图像块列表
        group_row: 组的行索引
        group_col: 组的列索引
        density_group_size: 密度分组大小

    Returns:
        list: 组内的patches列表
    """
    group_patches = []
    for marked in patches:
        m_row, m_col = marked[0], marked[1]
        if (
            group_row <= m_row < group_row + density_group_size
            and group_col <= m_col < group_col + density_group_size
        ):
            group_patches.append(marked)
    return group_patches


def is_group_brightness_valid(group_patches, img_rgb):
    """检查组的平均亮度是否有效

    Args:
        group_patches: 组内的patches列表
        img_rgb: RGB图像

    Returns:
        bool: 亮度是否有效
    """
    if img_rgb is None:
        return True
    
    group_brightness = []
    for patch in group_patches:
        _, _, x1, y1, x2, y2 = patch
        patch_img = img_rgb[y1:y2, x1:x2]
        if patch_img.size > 0:
            patch_color = np.mean(patch_img.reshape(-1, 3), axis=0)
            group_brightness.append(patch_color)

    if group_brightness:
        avg_brightness = np.mean(group_brightness, axis=0)
        return is_valid_brightness(avg_brightness)
    
    return True


def calculate_group_center(group_patches):
    """计算组的中心点

    Args:
        group_patches: 组内的patches列表

    Returns:
        dict: 包含'x'和'y'键的中心点坐标
    """
    group_centers = []
    for marked in group_patches:
        group_centers.append([marked[4], marked[5]])
    
    if group_centers:
        avg_center = np.mean(group_centers, axis=0)
        return {"x": int(avg_center[0]), "y": int(avg_center[1])}
    return None


def filter_processed_patches(patches, center_row, center_col, density_group_size):
    """过滤掉已处理的patches

    Args:
        patches: 标记的图像块列表
        center_row: 中心行索引
        center_col: 中心列索引
        density_group_size: 密度分组大小

    Returns:
        list: 过滤后的patches列表
    """
    return [
        marked
        for marked in patches
        if abs(marked[0] - center_row) > density_group_size
        or abs(marked[1] - center_col) > density_group_size
    ]


def calculate_density_centers(
    img_rgb, marked_patches, grid_size, density_group_size, top_n
):
    """计算密度聚类中心（逐步过滤临近区域 + 亮度筛选）

    Args:
        img_rgb: RGB图像
        marked_patches: 标记的图像块列表
        grid_size: 网格尺寸 (行, 列)
        density_group_size: 密度分组大小
        top_n: 返回前N个结果

    Returns:
        检测结果列表
    """
    result = []
    remaining_patches = marked_patches.copy()

    for _ in range(top_n):
        if len(remaining_patches) == 0:
            break
        
        density_scores = calculate_density_scores(
            remaining_patches, grid_size, density_group_size
        )

        if not density_scores:
            break

        sorted_groups = sorted(density_scores.items(), key=lambda x: x[1], reverse=True)

        found_valid_center = False
        best_group_row, best_group_col = None, None

        for group_pos, _ in sorted_groups:
            group_row, group_col = group_pos

            group_patches = get_patches_in_group(
                remaining_patches, group_row, group_col, density_group_size
            )

            if not group_patches:
                continue

            if not is_group_brightness_valid(group_patches, img_rgb):
                continue

            best_group_row = group_row
            best_group_col = group_col
            found_valid_center = True
            break

        if not found_valid_center:
            break

        group_patches = get_patches_in_group(
            remaining_patches, best_group_row, best_group_col, density_group_size
        )
        center = calculate_group_center(group_patches)
        if center:
            result.append(center)

        center_row = best_group_row + density_group_size // 2
        center_col = best_group_col + density_group_size // 2

        remaining_patches = filter_processed_patches(
            remaining_patches, center_row, center_col, density_group_size
        )

    result.sort(key=lambda p: (p["y"], p["x"]))
    return result


def detect_objects(
    image_path,
    patch_size=(3, 2),
    different_group_size=2,
    different_threshold=0.25,
    similarity_threshold=0.75,
    density_group_size=18,
    top_n=3,
):
    """检测图像中的物体位置

    Args:
        image_path: 图像文件路径
        patch_size: 图像块大小 (宽, 高)，默认(3, 2)
        different_group_size: 图像块颜色差异分组大小，默认2
        different_threshold: 图像块颜色差异阈值，默认0.25
        similarity_threshold: 已标记图像块进行9宫格扩展相似度阈值，默认0.75
        density_group_size: 所有标记图像块密度分组大小，默认18
        top_n: 按密度分组大小返回前N个检测结果，默认3

    Returns:
        tuple: (检测结果列表, 标记的图像块元组 (初始标记的图像块列表, 扩展标记的图像块列表), 网格尺寸, 已加载的图像BGR格式)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    grid_size = (height // patch_size[1], width // patch_size[0])
    # 初始标记图像块
    marked_initial_patches = mark_initial_patches(
        img_rgb, patch_size, grid_size, different_group_size, different_threshold
    )
    # 扩展标记图像块
    all_marked_patches, expanded_marked_patches = expand_marked_patches(
        marked_initial_patches,
        img_rgb,
        patch_size,
        grid_size,
        similarity_threshold,
    )
    # 计算密度中心
    positions = calculate_density_centers(
        img_rgb, all_marked_patches, grid_size, density_group_size, top_n
    )

    return positions, (marked_initial_patches, expanded_marked_patches), grid_size, img


def draw_patches(img, patches, color):
    """在图像上绘制patches

    Args:
        img: 图像（BGR格式）
        patches: 图像块列表，每个格式为 (row, col, x1, y1, x2, y2)
        color: 绘制颜色 (B, G, R)
    """
    if patches is None:
        return
    
    for patch in patches:
        _, _, x1, y1, x2, y2 = patch
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def draw_detection_points(img, positions):
    """在图像上绘制检测点

    Args:
        img: 图像（BGR格式）
        positions: 检测到的物体位置列表，每个元素包含'x'和'y'键
    """
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    for idx, pos in enumerate(positions):
        color = colors[idx % len(colors)]
        cv2.circle(img, (pos["x"], pos["y"]), 20, color, 3)
        print(f"  {idx}. x={pos['x']}, y={pos['y']}")


def visualize_results(
    img,
    positions,
    marked_patches,
    output_path="result.jpg",
):
    """可视化检测结果和网格标记

    Args:
        img: 已加载的图像（BGR格式）
        positions: 检测到的物体位置列表，每个元素包含'x'和'y'键
        marked_patches: 标记的图像块元组 (初始标记的图像块列表, 扩展标记的图像块列表)
                        每个图像块格式: (row, col, x1, y1, x2, y2)
        output_path: 输出图像路径，默认'result.jpg'
    """
    if img is None:
        return

    if marked_patches is not None:
        initial_marked_patches, expanded_marked_patches = marked_patches
    else:
        initial_marked_patches = None
        expanded_marked_patches = None

    draw_patches(img, initial_marked_patches, (0, 255, 0))
    draw_patches(img, expanded_marked_patches, (255, 0, 255))
    draw_detection_points(img, positions)
    
    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    resources_dir = "resources"
    result_dir = os.path.join(resources_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    jpg_files = glob.glob(os.path.join(resources_dir, "*.jpg"))
    patch_size = (3, 2)
    different_group_size = 2
    different_threshold = 0.25
    density_group_size = 18
    similarity_threshold = 0.75
    top_n = 3
    # 遍历所有JPG文件
    for image_path in jpg_files:
        (
            positions,
            marked_patches,
            grid_size,
            img,
        ) = detect_objects(
            image_path,
            patch_size,
            different_group_size,
            different_threshold,
            similarity_threshold,
            density_group_size,
            top_n,
        )
        # 可视化结果
        output_path = os.path.join(
            result_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_result.jpg",
        )
        visualize_results(
            img,
            positions,
            marked_patches,
            output_path,
        )
