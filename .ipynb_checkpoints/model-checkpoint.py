import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from tqdm import tqdm
import os
import numpy as np

class KGDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        head_id = self.entity2id[head]
        relation_id = self.relation2id[relation]
        tail_id = self.entity2id[tail]
        return head_id, relation_id, tail_id

class predDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation = self.triples[idx]
        head_id = self.entity2id[head]
        relation_id = self.relation2id[relation]
        return head_id, relation_id

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self,adj_x):
       
        #h = torch.matmul(adj_x)
        h = self.fc(adj_x)
        return F.relu(h)


class RetrieveAndReadFramework(nn.Module):
    def __init__(self, triples, entity2id, relation2id, entity_dim=128, relation_dim=128, lr=0.001, batch_size=32,
                 num_epochs=10, num_gnn_layers=5):
        super(RetrieveAndReadFramework, self).__init__()
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dataset = KGDataset(triples, entity2id, relation2id)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)

        self.entity_embedding = nn.Embedding(self.num_entities, entity_dim).to(self.device)
        self.relation_embedding = nn.Embedding(self.num_relations, relation_dim).to(self.device)
        self.gnn_layers = nn.ModuleList([GNNLayer(entity_dim, entity_dim) for _ in range(num_gnn_layers)]).to(
            self.device)
        self.fc = nn.Linear(entity_dim + relation_dim, self.num_entities).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
       
        G = nx.DiGraph()

        edge_list = [(entity2id[head], entity2id[tail], {'id': relation2id[relation]}) for head, relation, tail in triples]
        G.add_edges_from(edge_list)
            #print(f"Graph saved to {file_path}")
        
        """
        for head, relation, tail in triples:
            head_id = self.entity2id[head]
            tail_id = self.entity2id[tail]
            G.add_edge(head_id, tail_id, id=relation2id[relation])
        """
        self.G = G

        #self.entity_embedding = nn.DataParallel(self.entity_embedding)
        #self.relation_embedding = nn.DataParallel(self.relation_embedding)
        #self.gnn_layers = nn.ModuleList([nn.DataParallel(layer) for layer in self.gnn_layers])
        #self.fc = nn.DataParallel(self.fc)


    def create_adjacency_matrix(self, nodes, max_hops=1):
        all_nodes = set(nodes)
        
        #一阶
        for node in nodes:
            all_nodes = all_nodes | set(self.G.neighbors(node))
        """
        #二阶
        all_nodes=set(neighbor_nodes)
        for node in neighbor_nodes:
            all_nodes=all_nodes | set(self.G.neighbors(node))
        """
        #暂时只要一阶
        subgraph = self.G.subgraph(all_nodes)
        all_nodes = list(subgraph.nodes)
        adj = torch.eye(len(all_nodes), len(all_nodes))  # .to(self.device)
        all_embed = self.entity_embedding(torch.tensor(all_nodes))
       
        #adj=torch.tensor(nx.adjacency_matrix(subgraph).toarray())
        #adj = adj/adj.sum(dim=1, keepdim=True).clamp(min=1e-10)
    # 避免除以0，使用clamp将所有小于1e-10的元素设置为1e-10
        
        adj_coo = nx.to_scipy_sparse_array(subgraph).tocoo()
        # 转换为torch稀疏张量
        indices = torch.tensor(np.array([adj_coo.row, adj_coo.col]), dtype=torch.int)
        values = torch.tensor(adj_coo.data, dtype=torch.float)
        adj = torch.sparse_coo_tensor(indices, values, torch.Size(adj_coo.shape))
        #adj稀疏格式
        return all_nodes, all_embed, adj
        """
        for node in all_nodes:
            neighbors = list(subgraph.neighbors(node))
            for neighbor in neighbors:
                if neighbor in nodes:
                    adj[all_nodes.index(node)][all_nodes.index(neighbor)] = 1/len(neighbors)
        
        
        for node in nodes:
            neighbors = list(subgraph.neighbors(node))
            for neighbor in neighbors:
                adj[all_nodes.index(node)][all_nodes.index(neighbor)] = 1 / len(neighbors)
        """
    
    
    def train_model(self, va,train=None):
        if train!=None:
            self.train_dataset = KGDataset(train, self.entity2id, self.relation2id)
            self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            self.train()
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', leave=True)
            for head, relation, tail in progress_bar:
                head, relation, tail = head.to(self.device), relation.to(self.device), tail.to(self.device)
                self.optimizer.zero_grad()
                # nodes = list(set(head.tolist() + tail.tolist()))
                nodes = list(set(head.tolist()))
                all_nodes, all_embed, adj = self.create_adjacency_matrix(nodes)

                all_embed = all_embed.to(self.device)
                adj = adj.to(self.device)
                # print(adj.size(),"*",all_embed.size())

                # adj = adj.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
                for layer in self.gnn_layers:
                    adj_x=torch.matmul(adj, all_embed) if not adj.is_sparse else torch.sparse.mm(adj, all_embed)
                    all_embed = layer(torch.matmul(adj, all_embed))
                    # print(adj.size(),"*",all_embed.size())
                head_nodes = head.tolist()
                node_to_index = {node: index for index, node in enumerate(all_nodes)}
                indices = [node_to_index[node] for node in head_nodes]
                head_embed = all_embed[indices]
                relation_embed = self.relation_embedding(relation)
                embed = torch.cat((head_embed, relation_embed), dim=1)
                output = self.fc(embed)
                loss = self.criterion(output, tail)
                loss.backward()
                self.optimizer.step()

            acc = self.evaluate_model(va)
            tqdm.write(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}, Acc: {acc * 100 :.2f}%')
            progress_bar.refresh()
            if(len(self.train_dataset)>10000):
                self.save_model(f"./trained/Epoch{epoch+1}.pt")
            # print(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}, Acc: {acc*100 :.2f}%')

    def evaluate_model(self, val_triples):
        eval_dataset = KGDataset(val_triples, self.entity2id, self.relation2id)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        self.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for head, relation, tail in eval_loader:
                head, relation, tail = head.to(self.device), relation.to(self.device), tail.to(self.device)
                self.optimizer.zero_grad()
                # nodes = list(set(head.tolist() + tail.tolist()))
                nodes = list(set(head.tolist()))
                all_nodes, all_embed, adj = self.create_adjacency_matrix(nodes)

                all_embed = all_embed.to(self.device)
                adj = adj.to(self.device)
                # print(adj.size(),"*",all_embed.size())

                # adj = adj.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
                for layer in self.gnn_layers:
                    all_embed = layer(torch.matmul(adj, all_embed))
                    # print(adj.size(),"*",all_embed.size())
                head_nodes = head.tolist()
                node_to_index = {node: index for index, node in enumerate(all_nodes)}
                indices = [node_to_index[node] for node in head_nodes]
                head_embed = all_embed[indices]
                relation_embed = self.relation_embedding(relation)
                embed = torch.cat((head_embed, relation_embed), dim=1)
                output = self.fc(embed)
                predicted_tail = torch.argmax(output, dim=1)
                total_correct += (predicted_tail == tail).sum().item()
                total_samples += len(tail)
        accuracy = total_correct / total_samples
        # print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy

    def predict(self, triples, top_k=10):
        self.eval()
        dataset = predDataset(triples, self.entity2id, self.relation2id)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []
        id2entity = {v: k for k, v in self.entity2id.items()}

        with torch.no_grad():
            for head, relation in loader:
                head, relation= head.to(self.device), relation.to(self.device)
                self.optimizer.zero_grad()
                # nodes = list(set(head.tolist() + tail.tolist()))
                nodes = list(set(head.tolist()))
                all_nodes, all_embed, adj = self.create_adjacency_matrix(nodes)

                all_embed = all_embed.to(self.device)
                adj = adj.to(self.device)
                # print(adj.size(),"*",all_embed.size())

                # adj = adj.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
                for layer in self.gnn_layers:
                    all_embed = layer(torch.matmul(adj.to(self.device), all_embed.to(self.device)))
                    # print(adj.size(),"*",all_embed.size())
                head_nodes = head.tolist()
                node_to_index = {node: index for index, node in enumerate(all_nodes)}
                indices = [node_to_index[node] for node in head_nodes]
                head_embed = all_embed[indices]
                relation_embed = self.relation_embedding(relation)
                embed = torch.cat((head_embed, relation_embed), dim=1)
                output = self.fc(embed)
                _, top_indices = torch.topk(output, top_k, dim=1)

                for idx_list in top_indices:
                    
                    top_entities = [id2entity[idx.item()] for idx in idx_list]
                    results.append(top_entities)

        return results

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entity2id': self.entity2id,
            'relation2id': self.relation2id
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.entity2id = checkpoint['entity2id']
        self.relation2id = checkpoint['relation2id']
        print(f'Model loaded from {path}')





if  __name__=="__main__":
    import pandas as pd

    train_set = pd.read_csv("MyData/train.csv")[['Head', 'Relation', 'Tail']]
    va_set = pd.read_csv("MyData/vali.csv")[['Head', 'Relation', 'Tail']]
    train = [tuple(row) for row in train_set.itertuples(index=False)]
    va = [tuple(row) for row in va_set.itertuples(index=False)]
    test_set=pd.read_csv("MyData/test.csv")[['Head','Relation']]
    test=[tuple(row) for row in test_set.itertuples(index=False)]
    relation2text = pd.read_csv("MyData/relation.csv", encoding='UTF-8')[['Name', 'Text', 'ID']]
    entity2text = pd.read_csv("MyData/entity.csv", encoding='UTF-8')[['Name', 'Text', 'ID']]
    eDic = entity2text.set_index('Name')['ID'].to_dict()
    rDic = relation2text.set_index('Name')['ID'].to_dict()
    model = RetrieveAndReadFramework(train, eDic, rDic, entity_dim=16, relation_dim=16, lr=0.0001, batch_size=200,
                                     num_epochs=50, num_gnn_layers=5)

    model.train_model(train=train,va=va)
    model.evaluate_model(va)
    model.save_model(path="./trained/final.pt")
    result=model.predict(test)
    result = pd.DataFrame(result)
    test_set=pd.concat([test_set,result],axis=1)
    test_set.to_csv("result.tsv",header=None,sep='\t',index=False)
    