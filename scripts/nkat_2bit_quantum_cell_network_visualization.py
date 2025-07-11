#!/usr/bin/env python3
"""
NKAT理論：2ビット量子セルのネットワーク可視化
NKAT Theory: 2-Bit Quantum Cell Network Visualization

実装日時: 2025-01-18
作成者: NKAT Theory Research Group
目的: セル体積、配置、相互作用の3D可視化とテンソルネットワーク表現
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy.spatial.distance import pdist, squareform
from scipy.constants import hbar, c, physical_constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from datetime import datetime
from tqdm import tqdm

# 物理定数
planck_length = np.sqrt(hbar * physical_constants['Newtonian constant of gravitation'][0] / c**3)

class QuantumCellNetworkVisualizer:
    """2ビット量子セルネットワーク可視化器"""
    
    def __init__(self, cell_size_factor=2.35, grid_size=(5, 5, 5)):
        """
        初期化
        
        Args:
            cell_size_factor: プランク長に対するセルサイズの倍率
            grid_size: セルグリッドの次元 (x, y, z)
        """
        # 基本パラメータ
        self.cell_size = cell_size_factor * planck_length
        self.grid_size = grid_size
        self.total_cells = np.prod(grid_size)
        
        # 量子状態とPauli行列
        self.pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        
        # 2ビットセル基底状態
        self.basis_states = {
            '00': np.array([1, 0, 0, 0], dtype=complex),
            '01': np.array([0, 1, 0, 0], dtype=complex),
            '10': np.array([0, 0, 1, 0], dtype=complex),
            '11': np.array([0, 0, 0, 1], dtype=complex)
        }
        
        # セル位置とグラフの初期化
        self.cell_positions = self._generate_cell_positions()
        self.network_graph = None
        self.tensor_network = None
        
        print(f"🧊 2ビット量子セルネットワーク可視化器初期化")
        print(f"   グリッドサイズ: {grid_size}")
        print(f"   総セル数: {self.total_cells}")
        print(f"   セルサイズ: {self.cell_size/planck_length:.2f} × ℓ_P")
        print(f"   物理体積: {self.cell_size**3:.2e} m³")
    
    def _generate_cell_positions(self):
        """3Dグリッド上にセル位置を生成"""
        positions = {}
        cell_id = 0
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    # セル中心位置（プランク単位）
                    x = (i - self.grid_size[0]/2) * 2.35
                    y = (j - self.grid_size[1]/2) * 2.35  
                    z = (k - self.grid_size[2]/2) * 2.35
                    
                    positions[cell_id] = {
                        'coords': (x, y, z),
                        'grid_index': (i, j, k),
                        'state': np.random.choice(['00', '01', '10', '11']),
                        'volume': self.cell_size**3,
                        'info_content': 2.0  # 2ビット
                    }
                    cell_id += 1
        
        return positions
    
    def create_network_graph(self):
        """NetworkXグラフでセル間ネットワークを構築"""
        print("📊 ネットワークグラフ構築中...")
        
        G = nx.Graph()
        
        # ノード追加（各セル）
        for cell_id, cell_data in self.cell_positions.items():
            G.add_node(cell_id, 
                      pos=cell_data['coords'],
                      state=cell_data['state'],
                      volume=cell_data['volume'],
                      info_content=cell_data['info_content'])
        
        # エッジ追加（隣接セル間の相互作用）
        edge_weights = []
        
        for cell1 in tqdm(self.cell_positions.keys(), desc="セル間相互作用計算"):
            pos1 = np.array(self.cell_positions[cell1]['coords'])
            
            for cell2 in self.cell_positions.keys():
                if cell1 >= cell2:  # 重複回避
                    continue
                
                pos2 = np.array(self.cell_positions[cell2]['coords'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # 最近接および次近接のみ接続
                if distance < 4.0:  # プランク単位
                    # 量子相関強度を計算
                    state1 = self.cell_positions[cell1]['state']
                    state2 = self.cell_positions[cell2]['state']
                    correlation = self._calculate_quantum_correlation(state1, state2)
                    
                    # 非可換効果による重み
                    nc_weight = self._noncommutative_coupling(distance)
                    total_weight = correlation * nc_weight
                    
                    G.add_edge(cell1, cell2, 
                              weight=total_weight,
                              distance=distance,
                              correlation=correlation,
                              nc_coupling=nc_weight)
                    edge_weights.append(total_weight)
        
        self.network_graph = G
        print(f"   ノード数: {G.number_of_nodes()}")
        print(f"   エッジ数: {G.number_of_edges()}")
        print(f"   平均重み: {np.mean(edge_weights):.4f}")
        
        return G
    
    def _calculate_quantum_correlation(self, state1, state2):
        """2つのセル状態間の量子相関を計算"""
        s1 = self.basis_states[state1]
        s2 = self.basis_states[state2]
        
        # Fidelity（忠実度）計算
        fidelity = np.abs(np.vdot(s1, s2))**2
        
        # 相関が強いほど高い値
        return 1.0 - fidelity + 0.1  # 最小値保証
    
    def _noncommutative_coupling(self, distance):
        """非可換効果による距離依存結合強度"""
        theta = (2.35 * planck_length)**2  # 非可換パラメータ
        
        # 1/r² 型のクーロン的相互作用 + 非可換補正
        classical_coupling = 1.0 / (distance**2 + 0.1)
        nc_correction = theta / (planck_length**2 * distance**4 + theta)
        
        return classical_coupling * (1 + nc_correction)
    
    def create_tensor_network_representation(self):
        """テンソルネットワーク表現の構築"""
        print("🕸️  テンソルネットワーク表現構築中...")
        
        tensor_network = {}
        
        for cell_id, cell_data in self.cell_positions.items():
            # 各セルを4次元テンソルとして表現
            state_vector = self.basis_states[cell_data['state']]
            
            # テンソルインデックスの生成
            neighbors = list(self.network_graph.neighbors(cell_id)) if self.network_graph else []
            
            # セルテンソルの構築（状態 × 隣接接続）
            tensor_dims = [4]  # 4次元状態空間
            for neighbor in neighbors[:6]:  # 最大6近傍
                tensor_dims.append(2)  # 各接続は2次元
            
            # ランダム初期化（実際には物理的制約から決定）
            tensor_shape = tuple(tensor_dims)
            real_part = np.random.randn(*tensor_shape) * 0.1
            imag_part = np.random.randn(*tensor_shape) * 0.1
            tensor_data = real_part + 1j * imag_part
            
            # 状態に応じた重み設定
            state_index = ['00', '01', '10', '11'].index(cell_data['state'])
            tensor_data[state_index] *= 10.0  # 主状態の強化
            
            tensor_network[cell_id] = {
                'tensor': tensor_data,
                'indices': ['state'] + [f'bond_{n}' for n in neighbors[:6]],
                'dims': tensor_dims,
                'neighbors': neighbors
            }
        
        self.tensor_network = tensor_network
        print(f"   テンソル数: {len(tensor_network)}")
        print(f"   平均結合次数: {np.mean([len(t['neighbors']) for t in tensor_network.values()]):.2f}")
        
        return tensor_network
    
    def visualize_3d_cell_network(self):
        """3Dセルネットワークの可視化（Plotly）"""
        print("🎨 3Dネットワーク可視化中...")
        
        if self.network_graph is None:
            self.create_network_graph()
        
        # ノード位置とデータ準備
        node_positions = np.array([self.cell_positions[node]['coords'] 
                                  for node in self.network_graph.nodes()])
        
        node_states = [self.cell_positions[node]['state'] 
                      for node in self.network_graph.nodes()]
        
        node_volumes = [self.cell_positions[node]['volume'] 
                       for node in self.network_graph.nodes()]
        
        # エッジデータ準備
        edge_x, edge_y, edge_z = [], [], []
        edge_weights = []
        
        for edge in self.network_graph.edges(data=True):
            pos1 = self.cell_positions[edge[0]]['coords']
            pos2 = self.cell_positions[edge[1]]['coords']
            
            edge_x.extend([pos1[0], pos2[0], None])
            edge_y.extend([pos1[1], pos2[1], None])
            edge_z.extend([pos1[2], pos2[2], None])
            edge_weights.append(edge[2]['weight'])
        
        # 状態カラーマッピング
        state_colors = {'00': 'blue', '01': 'green', '10': 'red', '11': 'purple'}
        node_colors = [state_colors[state] for state in node_states]
        
        # Plotly 3Dプロット作成
        fig = go.Figure()
        
        # エッジ（セル間接続）
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            name='Cell Connections',
            hoverinfo='none'
        ))
        
        # ノード（量子セル）
        fig.add_trace(go.Scatter3d(
            x=node_positions[:, 0],
            y=node_positions[:, 1], 
            z=node_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                opacity=0.8,
                symbol='circle'
            ),
            text=[f'Cell {i}<br>State: {state}<br>Volume: {vol:.2e} m³' 
                  for i, (state, vol) in enumerate(zip(node_states, node_volumes))],
            hovertemplate='%{text}<extra></extra>',
            name='Quantum Cells'
        ))
        
        # レイアウト設定
        fig.update_layout(
            title='2-Bit Quantum Cell Network in 3D Space',
            scene=dict(
                xaxis_title='X (Planck Units)',
                yaxis_title='Y (Planck Units)',
                zaxis_title='Z (Planck Units)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def visualize_cell_volume_distribution(self):
        """セル体積分布の可視化"""
        print("📊 セル体積分布可視化中...")
        
        volumes = [cell['volume'] for cell in self.cell_positions.values()]
        states = [cell['state'] for cell in self.cell_positions.values()]
        
        # 状態別体積分布
        fig = px.histogram(
            x=volumes,
            color=states,
            title='Cell Volume Distribution by Quantum State',
            labels={'x': 'Volume (m³)', 'color': 'Quantum State'},
            nbins=20
        )
        
        fig.update_layout(
            xaxis_title='Cell Volume (m³)',
            yaxis_title='Count',
            width=800,
            height=500
        )
        
        return fig
    
    def visualize_distance_correlation_matrix(self):
        """セル間距離-相関マトリックスの可視化"""
        print("🔍 距離-相関マトリックス可視化中...")
        
        if self.network_graph is None:
            self.create_network_graph()
        
        # 距離マトリックス計算
        positions = [self.cell_positions[i]['coords'] for i in range(self.total_cells)]
        distance_matrix = squareform(pdist(positions))
        
        # 相関マトリックス計算
        correlation_matrix = np.zeros((self.total_cells, self.total_cells))
        
        for i in range(self.total_cells):
            for j in range(self.total_cells):
                if i != j:
                    state_i = self.cell_positions[i]['state']
                    state_j = self.cell_positions[j]['state']
                    correlation_matrix[i, j] = self._calculate_quantum_correlation(state_i, state_j)
        
        # ヒートマップ作成
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Distance Matrix', 'Correlation Matrix'],
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        fig.add_trace(
            go.Heatmap(z=distance_matrix, colorscale='Blues', name='Distance'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(z=correlation_matrix, colorscale='Reds', name='Correlation'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Distance vs Correlation Matrix Analysis',
            width=1200,
            height=500
        )
        
        return fig
    
    def create_tensor_network_diagram(self):
        """テンソルネットワーク図の作成"""
        print("🕸️  テンソルネットワーク図作成中...")
        
        if self.tensor_network is None:
            self.create_tensor_network_representation()
        
        # 2Dレイアウトでテンソルネットワーク可視化
        G_tensor = nx.Graph()
        
        # テンソルノード追加
        for tensor_id, tensor_data in self.tensor_network.items():
            G_tensor.add_node(f'T{tensor_id}', 
                            tensor_id=tensor_id,
                            bond_dim=len(tensor_data['neighbors']))
        
        # テンソル間結合追加
        for tensor_id, tensor_data in self.tensor_network.items():
            for neighbor_id in tensor_data['neighbors']:
                if neighbor_id in self.tensor_network:
                    G_tensor.add_edge(f'T{tensor_id}', f'T{neighbor_id}')
        
        # レイアウト計算
        pos = nx.spring_layout(G_tensor, k=2, iterations=50)
        
        # Plotly ネットワーク図
        edge_x, edge_y = [], []
        for edge in G_tensor.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in G_tensor.nodes()]
        node_y = [pos[node][1] for node in G_tensor.nodes()]
        
        fig = go.Figure()
        
        # エッジ
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            name='Tensor Bonds'
        ))
        
        # ノード
        node_info = [f'Tensor {node}<br>Bond Dimension: {G_tensor.nodes[node]["bond_dim"]}' 
                     for node in G_tensor.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=15, color='lightblue', line=dict(width=2, color='black')),
            text=[node.replace('T', '') for node in G_tensor.nodes()],
            textposition='middle center',
            hovertemplate='%{text}<extra></extra>',
            name='Tensors'
        ))
        
        fig.update_layout(
            title='Tensor Network Diagram',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig
    
    def generate_comprehensive_dashboard(self):
        """包括的ダッシュボードの生成"""
        print("🎛️  包括的ダッシュボード生成中...")
        
        # 全ての可視化を実行
        network_3d = self.visualize_3d_cell_network()
        volume_dist = self.visualize_cell_volume_distribution()
        distance_corr = self.visualize_distance_correlation_matrix()
        tensor_diagram = self.create_tensor_network_diagram()
        
        # HTMLダッシュボード作成
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NKAT 2-Bit Quantum Cell Network Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .header {{
                    text-align: center;
                    color: #333;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧊 NKAT 2-Bit Quantum Cell Network Visualization</h1>
                <p>時空の離散化：2ビット量子セル理論の包括的可視化</p>
                <p>生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>🌌 3D Quantum Cell Network</h2>
                <div id="network3d"></div>
            </div>
            
            <div class="grid">
                <div class="section">
                    <h3>📊 Volume Distribution</h3>
                    <div id="volumedist"></div>
                </div>
                <div class="section">
                    <h3>🕸️ Tensor Network</h3>
                    <div id="tensornet"></div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Distance-Correlation Analysis</h2>
                <div id="distcorr"></div>
            </div>
            
            <script>
                // Plotly graphs would be embedded here individually
                console.log('Dashboard loaded');
            </script>
        </body>
        </html>
        """
        
        return {
            'network_3d': network_3d,
            'volume_dist': volume_dist,
            'distance_corr': distance_corr,
            'tensor_diagram': tensor_diagram,
            'dashboard_html': dashboard_html
        }
    
    def save_all_visualizations(self):
        """全ての可視化結果の保存"""
        print("💾 可視化結果保存中...")
        
        # 出力ディレクトリ作成
        os.makedirs('Results/visualizations/2bit_cells', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 包括的ダッシュボード生成
        dashboard = self.generate_comprehensive_dashboard()
        
        # 各図を個別保存
        save_files = {}
        
        # 3Dネットワーク
        network_file = f'Results/visualizations/2bit_cells/3d_network_{timestamp}.html'
        dashboard['network_3d'].write_html(network_file)
        save_files['3d_network'] = network_file
        
        # 体積分布
        volume_file = f'Results/visualizations/2bit_cells/volume_distribution_{timestamp}.html'
        dashboard['volume_dist'].write_html(volume_file)
        save_files['volume_distribution'] = volume_file
        
        # 距離相関
        distance_file = f'Results/visualizations/2bit_cells/distance_correlation_{timestamp}.html'
        dashboard['distance_corr'].write_html(distance_file)
        save_files['distance_correlation'] = distance_file
        
        # テンソルネットワーク
        tensor_file = f'Results/visualizations/2bit_cells/tensor_network_{timestamp}.html'
        dashboard['tensor_diagram'].write_html(tensor_file)
        save_files['tensor_network'] = tensor_file
        
        # 統合ダッシュボード
        dashboard_file = f'Results/visualizations/2bit_cells/comprehensive_dashboard_{timestamp}.html'
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard['dashboard_html'])
        save_files['dashboard'] = dashboard_file
        
        # ネットワーク統計保存
        stats_file = f'Results/json/network_statistics_{timestamp}.json'
        network_stats = self._calculate_network_statistics()
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(network_stats, f, indent=2, ensure_ascii=False, default=str)
        save_files['statistics'] = stats_file
        
        print(f"📁 保存完了:")
        for viz_type, filepath in save_files.items():
            print(f"   {viz_type}: {filepath}")
        
        return save_files
    
    def _calculate_network_statistics(self):
        """ネットワーク統計の計算"""
        if self.network_graph is None:
            return {}
        
        G = self.network_graph
        
        stats = {
            'basic_stats': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_connected(G)
            },
            'centrality_measures': {
                'avg_degree_centrality': np.mean(list(nx.degree_centrality(G).values())),
                'avg_betweenness_centrality': np.mean(list(nx.betweenness_centrality(G).values())),
                'avg_closeness_centrality': np.mean(list(nx.closeness_centrality(G).values()))
            },
            'physical_properties': {
                'total_volume': float(self.total_cells * self.cell_size**3),
                'avg_cell_distance': float(np.mean([data['distance'] for _, _, data in G.edges(data=True)])),
                'avg_quantum_correlation': float(np.mean([data['correlation'] for _, _, data in G.edges(data=True)])),
                'avg_nc_coupling': float(np.mean([data['nc_coupling'] for _, _, data in G.edges(data=True)]))
            },
            'quantum_information': {
                'total_information_bits': float(self.total_cells * 2),
                'entropy_per_cell': float(2 * np.log(2)),
                'total_entropy': float(self.total_cells * 2 * np.log(2))
            }
        }
        
        if nx.is_connected(G):
            stats['path_properties'] = {
                'avg_shortest_path_length': nx.average_shortest_path_length(G),
                'diameter': nx.diameter(G),
                'radius': nx.radius(G)
            }
        
        return stats

def main():
    """メイン可視化実行"""
    print("🎨 NKAT理論：2ビット量子セルネットワーク可視化開始")
    print("="*70)
    
    # 可視化器初期化
    visualizer = QuantumCellNetworkVisualizer(
        cell_size_factor=2.35,
        grid_size=(4, 4, 4)  # 64セルの3Dグリッド
    )
    
    # ネットワーク構築
    visualizer.create_network_graph()
    
    # テンソルネットワーク表現
    visualizer.create_tensor_network_representation()
    
    # 全可視化保存
    saved_files = visualizer.save_all_visualizations()
    
    # 最終レポート
    print("\n" + "="*70)
    print("🏆 2ビット量子セルネットワーク可視化完了")
    print("="*70)
    
    print(f"🧊 ネットワーク特性:")
    print(f"   総セル数: {visualizer.total_cells}")
    print(f"   セルサイズ: {visualizer.cell_size/planck_length:.2f} × ℓ_P")
    print(f"   ネットワーク密度: {nx.density(visualizer.network_graph):.4f}")
    
    print(f"\n📁 生成ファイル:")
    for file_type, path in saved_files.items():
        print(f"   {file_type}: {path}")
    
    print(f"\n🌟 主要可視化:")
    print(f"   🌌 3D空間でのセル配置とネットワーク構造")
    print(f"   📊 量子状態別セル体積分布")
    print(f"   🔍 セル間距離と量子相関の関係")
    print(f"   🕸️  テンソルネットワーク図式表現")
    
    print(f"\n✅ 2ビット量子セル可視化システム準備完了！")
    print(f"📖 統合ダッシュボード: {saved_files['dashboard']}")

if __name__ == "__main__":
    main() 