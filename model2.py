import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
from tqdm import tqdm

class KGDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def len(self):
        return len(self.triples)

    def get(self, idx):
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

    def len(self):
        return len(self.triples)

    def get(self, idx):
        head, relation = self.triples[idx]
        head_id = self.entity2id[head]
        relation_id = self.relation2id[relation]
        return head_id, relation_id

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
        self.gnn_layers = nn.ModuleList([GCNConv(entity_dim, entity_dim) for _ in range(num_gnn_layers)]).to(self.device)
        self.fc = nn.Linear(entity_dim + relation_dim, self.num_entities).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def create_graph_data(self, nodes, triples):
        edge_index = []
        edge_attr = []
        for head, relation, tail in triples:
            head_id = self.entity2id[head]
            relation_id = self.relation2id[relation]
            tail_id = self.entity2id[tail]
            if head_id in nodes and tail_id in nodes:
                edge_index.append([head_id, tail_id])
                edge_attr.append(relation_id)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        return Data(x=self.entity_embedding.weight[nodes], edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    def train_model(self, train_triples, val_triples):
        for epoch in range(self.num_epochs):
            self.train()
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', leave=True)
            for head, relation, tail in progress_bar:
                head, relation, tail = head.to(self.device), relation.to(self.device), tail.to(self.device)
                self.optimizer.zero_grad()
                nodes = list(set(head.tolist()))
                data = self.create_graph_data(nodes, train_triples).to(self.device)
                all_embed = self.forward(data)
                head_embed = all_embed[head]
                relation_embed = self.relation_embedding(relation)
                embed = torch.cat((head_embed, relation_embed), dim=1)
                output = self.fc(embed)
                loss = self.criterion(output, tail)
                loss.backward()
                self.optimizer.step()

            acc = self.evaluate_model(val_triples)
            tqdm.write(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}, Acc: {acc * 100 :.2f}%')
            progress_bar.refresh()

    def evaluate_model(self, val_triples):
        eval_dataset = KGDataset(val_triples, self.entity2id, self.relation2id)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        self.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for head, relation, tail in eval_loader:
                head, relation, tail = head.to(self.device), relation.to(self.device), tail.to(self.device)
                nodes = list(set(head.tolist()))
                data = self.create_graph_data(nodes, val_triples).to(self.device)
                all_embed = self.forward(data)
                head_embed = all_embed[head]
                relation_embed = self.relation_embedding(relation)
                embed = torch.cat((head_embed, relation_embed), dim=1)
                output = self.fc(embed)
                predicted_tail = torch.argmax(output, dim=1)
                total_correct += (predicted_tail == tail).sum().item()
                total_samples += len(tail)
        accuracy = total_correct / total_samples
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
                nodes = list(set(head.tolist()))
                data = self.create_graph_data(nodes, []).to(self.device)
                all_embed = self.forward(data)
                head_embed = all_embed[head]
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

if __name__ == "__main__":
    train_set = pd.read_csv("MyData/train.csv")[['Head', 'Relation', 'Tail']]
    va_set = pd.read_csv("MyData/vali.csv")[['Head', 'Relation', 'Tail']]
    train = [tuple(row) for row in train_set.itertuples(index=False)]
    va = [tuple(row) for row in va_set.itertuples(index=False)]
    test_set = pd.read_csv("MyData/test.csv")[['Head', 'Relation']]
    test = [tuple(row) for row in test_set.itertuples(index=False)]
    relation2text = pd.read_csv("MyData/relation.csv", encoding='UTF-8')[['Name', 'Text', 'ID']]
    entity2text = pd.read_csv("MyData/entity.csv", encoding='UTF-8')[['Name', 'Text', 'ID']]
    eDic = entity2text.set_index('Name')['ID'].to_dict()
    rDic = relation2text.set_index('Name')['ID'].to_dict()
    model = RetrieveAndReadFramework(train, eDic, rDic, entity_dim=16, relation_dim=16, lr=0.0001, batch_size=200,
                                     num_epochs=50, num_gnn_layers=5)

    model.train_model(train_triples=train[:500], val_triples=va)
    model.evaluate_model(va)
    model.save_model(path="./trained/final.pt")
    result = model.predict(test)
    result = pd.DataFrame(result)
    test_set = pd.concat([test_set, result], axis=1)
    test_set.to_csv("result.tsv", header=None, sep='\t', index=False)
