import os
import heapq
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def calc_freq(image):
    freq = defaultdict(int)
    for pixel in image:
        freq[pixel] += 1
    return freq

def build_huffman_tree(freq):
    heap = [Node(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(root):
    codes = {}
    def helper(node, code):
        if node is not None:
            if node.char is not None:
                codes[node.char] = code
            helper(node.left, code + '0')
            helper(node.right, code + '1')
    helper(root, '')
    return codes

def huffman_encode(image, codes):
    return ''.join(codes[pixel] for pixel in image)

def huffman_decode(encoded, root):
    decoded = []
    node = root
    for bit in encoded:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.char is not None:
            decoded.append(node.char)
            node = root
    return decoded


def build_graph(node, parent=None, G=None):
    if G is None:
        G = nx.DiGraph()
    if parent is not None:
        G.add_edge(parent, node, weight=node.freq)
    if node.left is not None:
        G = build_graph(node.left, node, G)
    if node.right is not None:
        G = build_graph(node.right, node, G)
    return G

# 绘制哈夫曼树
def draw_huffman_tree(root):
    G = build_graph(root)
    pos = nx.spring_layout(G, k=0.35, scale=10)  # 修改节点间距
    plt.figure(figsize=(20, 20), dpi=300)  # 增加图像分辨率
    nx.draw_networkx(G, pos, with_labels=False, node_size=50, linewidths=0.5, edgecolors='black')  # 调整节点和边的样式
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, bbox=dict(alpha=0), font_size=8)  # 设置标签字号
    plt.show()

def main():
    for filename in os.listdir('./pic'):
        if filename.endswith('.jpg'):
            image = cv2.imread(os.path.join('pic', filename), cv2.IMREAD_GRAYSCALE)
            freq = calc_freq(image.flatten())
            root = build_huffman_tree(freq)
            draw_huffman_tree(root)
            codes = build_codes(root)
            encoded = huffman_encode(image.flatten(), codes)
            print(f"Encoded value for {filename}: {encoded}")
            with open(filename + '.bin', 'w') as f:
                f.write(encoded)
            with open(filename + '.bin', 'r') as f:
                encoded = f.read()
            decoded = huffman_decode(encoded, root)
            decoded_image = np.array(decoded).reshape(image.shape)
            cv2.imwrite('decoded_' + filename, decoded_image)

            # 画图
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            original_image = plt.imread(os.path.join('pic', filename))  # 读取原始图片文件
            plt.imshow(original_image)  # 显示原始图片
            plt.title('Original Image')
            plt.subplot(1, 2, 2)
            plt.imshow(decoded_image, cmap='gray')
            plt.title('Decoded Image')
            plt.show()

if __name__ == '__main__':
    main()