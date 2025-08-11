import jittor as jt
import numpy as np


def safe_convert_to_jittor(data, max_depth=10, current_depth=0):
    """安全地将数据转换为Jittor格式，避免无限递归"""
    if current_depth >= max_depth:
        return data  # 防止无限递归

    # 对原始/简单类型直接返回，避免不必要判定
    try:
        if data is None or isinstance(data, (int, float, bool, str, bytes)):
            return data
    except RecursionError:
        return data

    try:
        # 如果已经是Jittor变量，直接返回
        is_jt_var = False
        try:
            is_jt_var = isinstance(data, jt.Var)
        except RecursionError:
            # 如果在 __instancecheck__ 中递归，直接跳过转换
            return data
        if is_jt_var:
            return data

        # 如果是列表或元组，递归处理
        if isinstance(data, (list, tuple)):
            return [safe_convert_to_jittor(item, max_depth, current_depth + 1) for item in data]

        # 如果是字典，递归处理
        if isinstance(data, dict):
            return {key: safe_convert_to_jittor(value, max_depth, current_depth + 1) for key, value in data.items()}

        # 处理类似 DataContainer 的对象（有 data 属性）
        if hasattr(data, 'data') and not is_jt_var:
            # 检查是否是DataContainer类型
            if hasattr(data, 'stack') and hasattr(data, 'cpu_only'):
                # 这是DataContainer，提取其data属性
                inner = getattr(data, 'data')
                print(f"🔍 检测到DataContainer，提取data: {type(inner)}, shape: {getattr(inner, 'shape', 'unknown')}")
                # 递归处理内部数据
                return safe_convert_to_jittor(inner, max_depth, current_depth + 1)
            # 其他有data属性的对象
            inner = getattr(data, 'data')
            if isinstance(inner, list):
                converted_list = []
                for item in inner:
                    try:
                        # 避免递归调用 ensure_jittor_var，直接处理
                        if isinstance(item, jt.Var):
                            converted_list.append(item)
                        elif isinstance(item, np.ndarray):
                            converted_list.append(jt.array(item))
                        elif hasattr(item, 'cpu') and hasattr(item, 'numpy'):
                            converted_list.append(jt.array(item.detach().cpu().numpy()))
                        else:
                            converted_list.append(item)
                    except Exception:
                        converted_list.append(item)
                return converted_list
            elif isinstance(inner, np.ndarray):
                # 如果是numpy数组，直接转换
                return jt.array(inner)
            return safe_convert_to_jittor(inner, max_depth, current_depth + 1)

        # 如果是PyTorch张量或numpy数组，转换为Jittor
        if not is_jt_var:
            try:
                return ensure_jittor_var(data, "data")
            except Exception:
                return data

        # 如果是memoryview，转换为numpy数组
        if isinstance(data, memoryview):
            try:
                import numpy as np
                return jt.array(np.array(data))
            except Exception:
                return data

        # 其他情况，直接返回
        return data

    except Exception:
        # 静默处理错误，避免大量错误输出
        return data


def ensure_jittor_var(data, name="data", default_shape=None):
    """确保数据是Jittor张量，如果不是则转换，转换失败则返回默认值"""
    # 在函数开头导入numpy
    import numpy as np
    
    try:
        # 首先处理DataContainer类型
        if hasattr(data, 'data') and not isinstance(data, jt.Var):
            # 检查是否是DataContainer类型
            if hasattr(data, 'stack') and hasattr(data, 'cpu_only'):
                # 这是DataContainer，提取其data属性
                inner_data = data.data
                print(f"🔍 检测到DataContainer，提取data: {type(inner_data)}, shape: {getattr(inner_data, 'shape', 'unknown')}")
                # 递归处理内部数据
                return ensure_jittor_var(inner_data, name, default_shape)
        
        if isinstance(data, jt.Var):
            return data
        elif isinstance(data, (list, tuple)):
            # 如果是列表/元组，尝试转换为张量
            if len(data) == 0:
                if default_shape:
                    return jt.zeros(default_shape, dtype='float32')
                else:
                    return jt.zeros((0,), dtype='float32')
            else:
                # 尝试将列表转换为张量
                try:
                    # 先检查列表中的元素类型
                    if len(data) > 0:
                        first_item = data[0]
                        if isinstance(first_item, (list, tuple)):
                            # 如果是嵌套列表，需要特殊处理
                            try:
                                # 对于gt_bboxes，期望格式是 [[[x1,y1,x2,y2], ...], ...]
                                # 对于gt_labels，期望格式是 [[label1, label2, ...], ...]
                                if name == "gt_bboxes":
                                    # 处理gt_bboxes：展平所有batch的边界框
                                    all_bboxes = []
                                    for batch_bboxes in data:
                                        if isinstance(batch_bboxes, (list, tuple)):
                                            for bbox in batch_bboxes:
                                                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                                    all_bboxes.append(bbox)
                                    if all_bboxes:
                                        return jt.array(all_bboxes, dtype='float32')
                                    else:
                                        return jt.zeros((1, 4), dtype='float32')
                                elif name == "gt_labels":
                                    # 处理gt_labels：展平所有batch的标签
                                    all_labels = []
                                    for batch_labels in data:
                                        if isinstance(batch_labels, (list, tuple)):
                                            for label in batch_labels:
                                                if isinstance(label, (int, float)):
                                                    all_labels.append(int(label))
                                    if all_labels:
                                        return jt.array(all_labels, dtype='int32')
                                    else:
                                        return jt.zeros((1,), dtype='int32')
                                else:
                                    # 其他嵌套列表，尝试直接转换
                                    return jt.array(data)
                            except Exception as nested_error:
                                print(f"⚠️  嵌套列表处理失败 {name}: {nested_error}")
                                # 如果嵌套处理失败，返回默认值
                                if name == "gt_bboxes":
                                    return jt.zeros((1, 4), dtype='float32')
                                elif name == "gt_labels":
                                    return jt.zeros((1,), dtype='int32')
                                else:
                                    return jt.zeros((1,), dtype='float32')
                        else:
                            # 如果是简单列表，直接转换
                            return jt.array(data)
                    else:
                        return jt.array(data)
                except Exception as e:
                    print(f"⚠️  列表转张量失败 {name}: {e}")
                    if default_shape:
                        return jt.zeros(default_shape, dtype='float32')
                    else:
                        return jt.zeros((1,), dtype='float32')
        elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):
            # PyTorch张量
            return jt.array(data.detach().cpu().numpy())
        elif isinstance(data, np.ndarray):
            # NumPy数组
            return jt.array(data)
        else:
            # 其他类型，尝试转换为numpy再转为Jittor
            try:
                np_data = np.array(data)
                return jt.array(np_data)
            except Exception as e:
                print(f"⚠️  类型转换失败 {name}: {e}")
                if default_shape:
                    return jt.zeros(default_shape, dtype='float32')
                else:
                    return jt.zeros((1,), dtype='float32')
    except Exception as e:
        print(f"⚠️  确保Jittor张量失败 {name}: {e}")
        if default_shape:
            return jt.zeros(default_shape, dtype='float32')
        else:
            return jt.zeros((1,), dtype='float32')


def safe_sum(tensor, dim=None, name="tensor"):
    """安全的sum操作，确保输入是Jittor张量"""
    try:
        safe_tensor = ensure_jittor_var(tensor, name)
        if dim is not None:
            return safe_tensor.sum(dim=dim)
        else:
            return safe_tensor.sum()
    except Exception as e:
        print(f"⚠️  sum操作失败 {name}: {e}")
        # 返回零张量
        if dim is not None:
            return jt.zeros((1,), dtype='float32')
        else:
            return jt.zeros((1,), dtype='float32')