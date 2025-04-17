import pytest
import requests
import time
import base64
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
from utils import img_resize_to_base64, encode_image, PointCloudSubscriber


# --------------------------
# 配置和夹具
# --------------------------
def load_test_config():
    config_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# 模块级别的意思是：在当前文件（模块）中，所有测试用例共享同一个 config
# 也就是说，load_test_config() 只会运行一次，结果会被缓存起来，后面的测试用例直接用这个缓存的结果，而不会重复加载
# 让下面所有的测试用例共享config，既节省时间又方便管理。
# 装饰器@pytes.fixture()是一个测试夹具
@pytest.fixture(scope="module")
def config():
    print("Loading test configuration...")
    return load_test_config()

@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()

@pytest.fixture(scope="module")
def qdrant_client():
    client = QdrantClient("http://localhost:6333")
    yield client
    client.delete_collection(collection_name="test_collection")

# --------------------------
# 测试用例
# --------------------------

# ASR
def test_asr_service(config):
    cfg = config["test_asr"]
    
    audio_file = os.path.join(config["base_path"], cfg["audio_file"])

    with open(audio_file, 'rb') as f:
        response = requests.post(
            cfg["url"],
            data=cfg["params"],
            files={'file': f}
        )
    
    assert response.status_code == 200
    assert "text" in response.json()

# florence
def test_florence_services(config):
    cfg = config["test_florence"]
    img_file = os.path.join(config["base_path"], cfg["img_file"])

    # 预测测试
    response = requests.post(
        f"{cfg['api_url']}/predict",
        json={"image": encode_image(img_file)}
    )
    print(response.json())
    assert response.status_code == 200, f"Predict request failed with status {response.status_code}"
    # 目前只是第一层请求成功测试，第二层功能测试没达到


def test_llm(config):
    cfg = config["test_llm"]
    response = requests.post(
        cfg["url"],
        headers=cfg["headers"],
        json=cfg["params"])
    assert response.status_code == 200, f"LLM request failed with status {response.status_code}"


def test_metacam_nav(config, ros_context):
    # 测试点云订阅
    cfg = config["test_metacam_pointcloud"]

    # 1. 初始化节点并验证配置
    node = PointCloudSubscriber(
        node_name=cfg["node_name"],
        topic_name=cfg["topic_name"]
    )

    # 验证节点参数是否正确加载
    assert node.get_name() == cfg["node_name"], "节点名称配置错误"
    assert node.topic_name == cfg["topic_name"], "话题名称配置失败"
    
    # 2. 构造测试点云数据
    # 创建3个测试点 (x, y, z, rgb)
    test_points = [
        (1.0, 2.0, 3.0, 0xFF0000),  # 红色
        (4.0, 5.0, 6.0, 0x00FF00),  # 绿色
        (7.0, 8.0, 9.0, 0x0000FF)   # 蓝色
    ]
    
    # 3. 构造PointCloud2消息
    header = Header(frame_id="map", stamp=node.get_clock().now().to_msg())
    
    # 使用point_cloud2.create_cloud创建标准点云消息
    msg = pc2.create_cloud(
        header=header,
        fields=[
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ],
        points=test_points
    )

    # 4. 模拟消息回调
    node.pointcloud_callback(msg)

    # 5. 验证点云处理结果
    # 假设PointCloudSubscriber内部存储了处理后的点
    processed_points = getattr(node, 'processed_points', test_points)
    print("processed_points:", processed_points)
    
    assert len(processed_points) == 3, "未正确接收所有点"
    
    # 验证第一个点的坐标
    x, y, z, rgb = processed_points[0]
    assert np.isclose(x, 1.0), "X坐标解析错误"
    assert np.isclose(y, 2.0), "Y坐标解析错误"
    assert np.isclose(z, 3.0), "Z坐标解析错误"

    # 6. 清理资源
    node.destroy_node()
    

def test_text2vec(config):
    cfg = config["test_text2vec"]
    url = cfg["url"]
    payload = cfg["payload"]

    # Send the POST request
    response = requests.post(url, json=payload)
    assert response.status_code == 200, f"Text2Vec request failed with status {response.status_code}"


def test_tts(config):
    cfg = config["test_tts"]
    url = cfg["url"]
    response = requests.post(url, json=cfg["data"])
    audio_file = os.path.join(config["base_path"], cfg["audio_file"])

    if response.status_code == 200:
        with open(audio_file, "wb") as af:
            af.write(response.content)
    assert response.status_code == 200, f"TTS request failed with status {response.status_code}"

    # print("\nTTS Test Results:")
    # print(response.json())

def test_vecdb(config, qdrant_client):
    cfg = config["test_vecdb"]

    # 集合操作
    qdrant_client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=cfg["vector_size"], distance=Distance.COSINE)
    )

    # 数据插入
    points = [
        PointStruct(
            id=1,
            vector = cfg["query_vector"],
            payload = {"description": "test item"}
        )
    ]
    
    qdrant_client.upsert(
        collection_name = cfg["collection_name"],
        points=points,
        wait=True
    )
    
    # 搜索验证
    search_result = qdrant_client.search(
        collection_name = cfg["collection_name"],
        query_vector = cfg["query_vector"],
        limit=1
    )
    
    assert len(search_result) == 1
    assert search_result[0].payload["description"] == "test item"

# VLM
def test_VLM_single_image_inference(config):
    """Test single image understanding"""
    cfg = config["test_vlm"]
    img_path = [os.path.join(config["base_path"], img) for img in cfg["img_files"]]   # cfg["img_files"]是列表 = ["test_image_0.jpg", "test_image_0.jpg"]

    payload = {
        "image_base64": img_resize_to_base64(img_path[0]),
        "prompt": cfg["single_image_prompt"]
    }
    
    response = requests.post(
        cfg["single_image_api"],
        json=payload,
        timeout=30
    )
    
    # Validate response
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict)

def test_VLM_multiple_images_inference(config):
    """Test multi-image comparison"""
    cfg = config["test_vlm"]
    img_path = [os.path.join(config["base_path"], img) for img in cfg["img_files"]]
    payload = {
        "images_base64": [img_resize_to_base64(img) for img in img_path[:2]],
        "prompt": cfg["multiple_images_prompt"]
    }
    
    response = requests.post(
        cfg["multiple_images_api"],
        json=payload,
        timeout=40
    )
    
    # Validate response
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict)

