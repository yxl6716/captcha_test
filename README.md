# Captcha_Test

简单识别图片验证码的图标大致位置（geetest极验v4为例）

本项目由个人思路+AI生成代码+调试优化，未识别图标朝向，测试图片数量有限，仅作为图像位置识别的思路参考

## 算法流程

1. **图像预处理**：读取图像并转换为 RGB 格式
2. **网格分割**：将图像划分为小的图像块
3. **分组并亮度+颜色差异阈值筛选初始标记**：在每个分组中识别与背景差异显著的图像块
4. **相邻网格亮度+颜色相似度筛选扩展标记**：基于相似度扩展标记区域
6. **计算所有标记的密度并按大小返回**：计算密度最高的区域作为目标物体中心
7. **可视化结果输出**：根据返回的物体坐标和标记的图像块，在原始图像上绘制矩形框并保存可视化图像

## 安装依赖

使用uv安装依赖

```bash
uv pip install -r requirements.txt
```
或使用pip安装依赖

```bash
pip install -r requirements.txt
```
### 依赖项

- `opencv-python` - 图像处理
- `numpy` - 数值计算

## 使用方法

### 基本用法

```python
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
```

### 参数说明

#### `detect_objects` 函数参数

- `image_path` (str): 图像文件路径
- `patch_size` (tuple): 图像块大小 (宽, 高)，默认 (3, 2)，即测试图片长宽300*200像素，则被分为100*100个图像块
- `different_group_size` (int): 颜色差异分组大小，默认 2，即2*2图像块分组，标记1个颜色差异最大的图像块
- `different_threshold` (float): 颜色差异阈值，默认 0.25，即分组中颜色差异最大的图像块与背景颜色差异大于0.25才被标记为目标物体
- `similarity_threshold` (float): 九宫格扩展颜色相似度阈值，默认 0.75，即每个图标的颜色相似度阈值
- `density_group_size` (int): 密度分组大小，默认 18，即18*18个图像块分组计算最大标记图像块的密度，18*3=54像素基本等于最大图标的外接矩形的长
- `top_n` (int): 密度分组返回前 N 个检测结果，默认 3

#### `visualize_results` 函数参数

- `img` (ndarray): 已加载的图像（BGR格式）
- `positions` (list): 检测到的物体位置列表，每个元素包含'x'和'y'键
- `marked_patches` (tuple): 标记的图像块元组 (初始标记的图像块列表, 扩展标记的图像块列表)
                        每个图像块格式: (row, col, x1, y1, x2, y2)
- `output_path` (str): 输出图像路径，默认'result.jpg'

## 可视化说明

生成的结果图像包含以下标记：

- **绿色框线**：初始标记的图像块
- **洋红色框线**：通过九宫格对初始标记图像块进行扩展标记的图像块
- **红色圆圈**：第一个检测到的物体中心
- **绿色圆圈**：第二个检测到的物体中心
- **蓝色圆圈**：第三个检测到的物体中心


## 示例

项目包含三个测试图像：

- `resources/test0.jpg`
- `resources/test1.jpg`
- `resources/test2.jpg`

对应的检测结果：

- `resources/result/test0_result.jpg`
- `resources/result/test1_result.jpg`
- `resources/result/test2_result.jpg`

## 许可证

该项目采用 [GNU General Public License v3.0](LICENSE) 许可证进行开源。