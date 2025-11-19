import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import requests
import time
from datetime import datetime, timedelta
from collections import defaultdict
import praw # For Reddit API
import transformers # For BERT tokenizer and model
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F # Import torch.nn.functional as F

# Base directory and model directory for deployment (no Colab/Drive paths)
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Streamlit page configuration must be the first Streamlit call
st.set_page_config(page_title="Bot Detection App", layout="wide")

# --- Configuration & Global Variables ---
# Determine device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
st.session_state.device = device # Store device in session state for access across reruns

# Output path for models and data (local models directory for deployment)
output_path = MODELS_DIR
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# Initialize BERT tokenizer and model globally, once
@st.cache_resource
def load_bert_components():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(st.session_state.device)
    bert_model.eval()
    return tokenizer, bert_model

tokenizer, bert_model = load_bert_components()

# --- 2. Memory-Efficient DataLoader Preparation (MMT, CTPP-GNN, BiLSTM-Att) ---
# Custom Dataset Class for BiLSTM-Att and MMT
class TextMetadataDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, metadata_cols, img_features_mmap=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.metadata_cols = metadata_cols
        self.img_features_mmap = img_features_mmap # None for BiLSTM-Att

        # Prepare the dataframe with one-hot encoded platforms and selected metadata for consistency
        platform_encoded = pd.get_dummies(self.dataframe['platform'], prefix='platform').astype(int)
        self.dataframe_prepared = pd.concat([self.dataframe, platform_encoded], axis=1)
        self.final_metadata_cols = [col for col in self.metadata_cols if col in self.dataframe_prepared.columns]

    def __len__(self):
        return len(self.dataframe_prepared)

    def __getitem__(self, idx):
        row = self.dataframe_prepared.iloc[idx]
        bio_text = str(row['bio_text']) if pd.notna(row['bio_text']) else ''
        metadata = row[self.final_metadata_cols].values.astype(np.float32)
        label = row['label'] if 'label' in row else -1 # Default label for inference

        img_features = None
        if self.img_features_mmap is not None:
            img_features = self.img_features_mmap[idx]

        return bio_text, metadata, img_features, label

# Custom Collate Function for BiLSTM-Att
def bilstm_collate_batch(batch):
    bio_texts = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch]
    labels = [item[3] for item in batch] # Assuming img_features is None for BiLSTM

    encoded_inputs = tokenizer(bio_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    metadata_tensor = torch.tensor(np.array(metadata_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'metadata': metadata_tensor,
        'labels': labels_tensor
    }

# Custom Collate Function for MMT
def mmt_collate_batch(batch):
    bio_texts = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch]
    img_features_list = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    encoded_inputs = tokenizer(bio_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    metadata_tensor = torch.tensor(np.array(metadata_list), dtype=torch.float32)
    img_features_tensor = torch.tensor(np.array(img_features_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'metadata': metadata_tensor,
        'img_features': img_features_tensor,
        'labels': labels_tensor
    }

# Custom Dataset and Collate for GCN
class GCNDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Ensure (N, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Custom Dataset and Collate for TGN
class TGNDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return feature vector, label, and its index (for synthetic timestamp/node ID)
        return self.features[idx], self.labels[idx], idx

def tgn_collate_fn(batch):
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    indices_list = [item[2] for item in batch]

    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32).unsqueeze(1)

    src_nodes = torch.tensor(indices_list, dtype=torch.long)
    dst_nodes = torch.tensor(indices_list, dtype=torch.long)
    t_events = torch.tensor(indices_list, dtype=torch.float32)
    messages = features_tensor # Node features serve as messages

    return {
        'features': features_tensor,
        'labels': labels_tensor,
        'src_nodes': src_nodes,
        'dst_nodes': dst_nodes,
        't_events': t_events,
        'messages': messages
    }

# Custom Dataset and Collate for CTPP-GNN
class CTPPDataset(torch.utils.data.Dataset):
    def __init__(self, timestamps, src_nodes, dst_nodes, edge_types, event_features, labels):
        self.timestamps = timestamps
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes
        self.edge_types = edge_types
        self.event_features = event_features
        self.labels = labels

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        return (
            self.timestamps[idx],
            self.src_nodes[idx],
            self.dst_nodes[idx],
            self.edge_types[idx],
            self.event_features[idx],
            self.labels[idx]
        )

def ctpp_collate_fn(batch):
    timestamps_list = [item[0] for item in batch]
    src_nodes_list = [item[1] for item in batch]
    dst_nodes_list = [item[2] for item in batch]
    edge_types_list = [item[3] for item in batch]
    event_features_list = [item[4] for item in batch]
    labels_list = [item[5] for item in batch]

    timestamps = torch.tensor(np.array(timestamps_list), dtype=torch.float32)
    src_nodes = torch.tensor(np.array(src_nodes_list), dtype=torch.long)
    dst_nodes = torch.tensor(np.array(dst_nodes_list), dtype=torch.long)
    edge_types = torch.tensor(np.array(edge_types_list), dtype=torch.long)
    event_features = torch.tensor(np.array(event_features_list), dtype=torch.float32)
    labels = torch.tensor(np.array(labels_list), dtype=torch.float32).unsqueeze(1)

    # For synthetic self-interactions, edge_index can be a simple self-loop
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    return {
        'timestamps': timestamps,
        'src_nodes': src_nodes,
        'dst_nodes': dst_nodes,
        'edge_types': edge_types,
        'event_features': event_features,
        'labels': labels,
        'edge_index': edge_index
    }

# Function to create sparse identity adjacency matrix (for GCN and CTPP-GNN)
def create_sparse_identity_adj(batch_size, device):
    indices = torch.arange(batch_size, device=device).unsqueeze(0).repeat(2, 1)
    values = torch.ones(batch_size, device=device, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, torch.Size([batch_size, batch_size]), device=device)
    return adj

# --- 3. Model Architectures ---

# CNN for BERT embeddings (used for XGBoost features, and then as input to GCN/TGN/CTPP-GNN)
class CNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Calculate output dimension after pooling. Assuming input_dim is sequence length if it were text.
        # But here input_dim=768 is the feature dimension, treated as a sequence of 1 (channel) and 768 (length).
        # After conv1 (kernel 3, pad 1), length is still 768. After pool (kernel 2), length is 768 // 2 = 384.
        self.fc = nn.Linear(hidden_dim * 384, 64) # Adjust for input_dim=768

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim (batch, 1, input_dim)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# BiLSTM with Attention Model
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_emb = nn.Linear(feature_dim, feature_dim, bias=False)
        self.context_vector = nn.Parameter(torch.rand(feature_dim))

    def forward(self, x, mask=None):
        et = self.features_emb(x)
        out = torch.sum(et * self.context_vector, dim=-1)
        out = torch.tanh(out)

        if mask is not None:
            current_sequence_length = out.shape[1]
            if mask.shape[1] > current_sequence_length:
                mask = mask[:, :current_sequence_length]
            out = out.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(out, dim=-1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights # RETURN WEIGHTS

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_metadata_features, output_dim=1, dropout_rate=0.5):
        super(BiLSTMAttentionClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_metadata_features = num_metadata_features

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.attention = Attention(feature_dim=2 * hidden_dim, step_dim=128)
        self.combined_feature_dim = (2 * hidden_dim) + num_metadata_features
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.combined_feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask, metadata):
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        context_vector, attention_weights = self.attention(lstm_output, mask=attention_mask) # CAPTURE ATTENTION WEIGHTS
        combined_features = torch.cat((context_vector, metadata), dim=1)
        combined_features = self.dropout(combined_features)
        combined_features = self.fc1(combined_features)
        combined_features = self.relu(combined_features)
        logits = self.classifier(combined_features)
        return logits, attention_weights # RETURN ATTENTION WEIGHTS

# GCN Model
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, features, adj):
        features_fp32 = features.to(torch.float32)
        support = torch.sparse.mm(adj, features_fp32)
        output = torch.mm(support, self.linear.weight.T)
        return output

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.5):
        super(GCNClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gcn_layers = nn.ModuleList()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.gcn_layers.append(GCNLayer(prev_dim, h_dim))
            prev_dim = h_dim

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(prev_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, features, adj):
        x = features
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, adj)
            x = self.relu(x)
            if i < len(self.gcn_layers) - 1:
                x = self.dropout(x)

        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# TGN Model
class MessageFunction(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MessageFunction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, src_features, dst_features, message_features):
        combined = torch.cat([src_features, dst_features, message_features], dim=-1)
        return self.mlp(combined)

class MemoryUpdate(nn.Module):
    def __init__(self, memory_dim, message_dim):
        super(MemoryUpdate, self).__init__()
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def forward(self, incoming_message, current_memory):
        return self.gru(incoming_message, current_memory)

class TemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, node_feat_dim, message_dim, memory_dim, hidden_dim, total_num_nodes, output_dim=1):
        super(TemporalGraphNetwork, self).__init__()
        self.input_dim = input_dim
        self.node_feat_dim = node_feat_dim
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.total_num_nodes = total_num_nodes

        self.node_feature_projection = nn.Linear(input_dim, node_feat_dim)
        self.message_function = MessageFunction(2 * node_feat_dim + input_dim, message_dim)
        self.memory_update = MemoryUpdate(memory_dim, message_dim)

        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.node_memories = nn.Parameter(torch.zeros(self.total_num_nodes, self.memory_dim, dtype=torch.float32))

    def forward(self, features, src_nodes, dst_nodes, t_events, messages):
        batch_size = features.shape[0]
        projected_features = self.node_feature_projection(features)

        current_src_memories = self.node_memories[src_nodes].to(features.dtype)
        current_dst_memories = self.node_memories[dst_nodes].to(features.dtype)

        event_messages = self.message_function(current_src_memories, current_dst_memories, messages)
        updated_memories = self.memory_update(event_messages, current_src_memories)

        self.node_memories.data[src_nodes] = updated_memories.data.to(torch.float32)

        logits = self.classifier(updated_memories)
        return logits

# MMT Model
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, dropout_rate=0.1):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers=num_layers)
        self.cls_token_processor = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        output = self.transformer_encoder(embedded, src_key_padding_mask=(attention_mask == 0))
        cls_representation = output[:, 0, :]
        return F.relu(self.cls_token_processor(cls_representation))

class ImageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout_rate=0.1):
        super(ImageEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, img_features):
        batch_size, seq_len, feat_dim = img_features.shape
        projected_features = self.input_projection(img_features)
        output = self.transformer_encoder(projected_features)
        output = output.permute(0, 2, 1)
        pooled_output = self.pool(output).squeeze(-1)
        return F.relu(pooled_output)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.mha(query=query, key=key, value=value, key_padding_mask=key_padding_mask)
        return self.norm(attn_output + query)

class MultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, text_embedding_dim, text_hidden_dim, text_num_heads, text_num_layers,
                 img_input_dim, img_hidden_dim, img_num_heads, img_num_layers,
                 metadata_input_dim, classifier_hidden_dim, output_dim=1, dropout_rate=0.1):
        super(MultimodalTransformer, self).__init__()

        self.text_encoder = TextEncoder(vocab_size, text_embedding_dim, text_hidden_dim, text_num_heads, text_num_layers, dropout_rate)
        self.image_encoder = ImageEncoder(img_input_dim, img_hidden_dim, img_num_heads, img_num_layers, dropout_rate)
        self.cross_attention = CrossAttention(query_dim=text_hidden_dim, key_dim=img_hidden_dim, value_dim=img_hidden_dim, num_heads=text_num_heads)

        combined_feature_dim = text_hidden_dim + metadata_input_dim
        self.fc1 = nn.Linear(combined_feature_dim, classifier_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(classifier_hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask, metadata, img_features):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(img_features)

        text_features_expanded = text_features.unsqueeze(1)
        image_features_expanded = image_features.unsqueeze(1)

        fused_features = self.cross_attention(query=text_features_expanded, key=image_features_expanded, value=image_features_expanded).squeeze(1)

        final_features = torch.cat((fused_features, metadata), dim=1)
        x = self.fc1(final_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# CTPP-GNN Model
class TPPIntensityModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(TPPIntensityModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        return F.softplus(self.mlp(features)).squeeze(-1)

class MultiNetworkGNN(nn.Module):
    def __init__(self, in_features, out_features, num_edge_types, num_layers=1):
        super(MultiNetworkGNN, self).__init__()
        self.num_edge_types = num_edge_types
        self.num_layers = num_layers
        self.gcn_type_networks = nn.ModuleList()

        for _ in range(num_edge_types):
            type_layers = nn.ModuleList()
            type_layers.append(GCNLayer(in_features, out_features))
            for _ in range(num_layers - 1):
                type_layers.append(GCNLayer(out_features, out_features))
            self.gcn_type_networks.append(type_layers)

    def forward(self, features, edge_types, adj):
        batch_size = features.shape[0]
        per_type_outputs = [None] * self.num_edge_types

        for etype in range(self.num_edge_types):
            gcn_layers_for_type = self.gcn_type_networks[etype]
            h_current = features
            for layer in gcn_layers_for_type:
                h_current = layer(h_current, adj)
                h_current = F.relu(h_current)
            per_type_outputs[etype] = h_current

        stacked_outputs = torch.stack(per_type_outputs, dim=0)
        batch_indices = torch.arange(batch_size, device=features.device)
        output_features = stacked_outputs[edge_types, batch_indices, :]
        output_features = output_features.view(batch_size, -1)
        return output_features

class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Aggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        return self.mlp(features)

class CTPPGNN(nn.Module):
    def __init__(self, input_dim, num_event_types, node_embedding_dim, gcn_hidden_dim,
                 tpp_hidden_dim, aggregator_hidden_dim, output_dim=1, gcn_num_layers=1,
                 dropout_rate=0.1):
        super(CTPPGNN, self).__init__()
        self.input_dim = input_dim
        self.num_event_types = num_event_types
        self.node_embedding_dim = node_embedding_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.tpp_hidden_dim = tpp_hidden_dim
        self.aggregator_hidden_dim = aggregator_hidden_dim
        self.output_dim = output_dim
        self.gcn_num_layers = gcn_num_layers

        self.node_feature_projection = nn.Linear(input_dim, node_embedding_dim)
        self.multi_network_gnn = MultiNetworkGNN(
            in_features=node_embedding_dim,
            out_features=gcn_hidden_dim,
            num_edge_types=num_event_types,
            num_layers=gcn_num_layers
        )
        self.tpp_intensity_module = TPPIntensityModule(
            input_dim=gcn_hidden_dim + input_dim,
            hidden_dim=tpp_hidden_dim,
            output_dim=1
        )
        self.aggregator = Aggregator(
            input_dim=gcn_hidden_dim,
            output_dim=aggregator_hidden_dim,
            hidden_dim=aggregator_hidden_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(aggregator_hidden_dim, aggregator_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(aggregator_hidden_dim // 2, output_dim)
        )

    def forward(self, event_features, timestamps, src_nodes, dst_nodes, edge_types, edge_index):
        batch_size = event_features.shape[0]
        node_embeddings = self.node_feature_projection(event_features)
        adj = create_sparse_identity_adj(batch_size, event_features.device)
        output_gnn = self.multi_network_gnn(node_embeddings, edge_types, adj)

        tpp_input = torch.cat([output_gnn, event_features], dim=-1)
        intensities = self.tpp_intensity_module(tpp_input)

        aggregated_features = self.aggregator(output_gnn)
        logits = self.classifier(aggregated_features)
        return logits


# --- Helper Functions for Data Preprocessing (for Inference) ---
def get_bert_embeddings_inference(texts, tokenizer, bert_model, device, max_seq_len=128):
    bert_model.eval()
    if not isinstance(texts, list):
        texts = [texts]
    # Truncate to max_seq_len for models that use BERT embeddings directly
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy() # CLS token

def get_cnn_features_inference(bert_embeddings, cnn_model, device):
    cnn_model.eval()
    bert_output_tensor = torch.tensor(bert_embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        cnn_features = cnn_model(bert_output_tensor).cpu().numpy()
    return cnn_features

def preprocess_metadata(metadata_dict, meta_cols_numerical, platform_cols_onehot):
    sample_df = pd.DataFrame([metadata_dict])
    for col in meta_cols_numerical:
        if col in sample_df.columns:
            sample_df[col] = np.log1p(sample_df[col].clip(0))
        else:
            sample_df[col] = 0.0

    platform_encoded = pd.get_dummies(sample_df['platform'], prefix='platform').astype(int)
    for p_col in platform_cols_onehot:
        if p_col not in platform_encoded.columns:
            platform_encoded[p_col] = 0

    # Ensure correct order and selection of metadata columns.
    # Dynamically build the final metadata columns list as expected by models.
    # For models using the smaller set of meta_cols_numerical_inference (XGBoost, GCN, TGN, CTPP-GNN)
    # The combined feature vector is 64 (CNN) + 9 (meta) + 2 (platform) = 75
    if len(meta_cols_numerical) == 9: # This indicates the smaller set of meta_cols_numerical_inference
        final_metadata_cols = [col for col in meta_cols_numerical if col in sample_df.columns] + \
                              [col for col in platform_cols_onehot if col in platform_encoded.columns]
    else: # This indicates the larger set of meta_cols_numerical_inference_mmt (12 columns + 2 platform = 14)
        final_metadata_cols = [col for col in meta_cols_numerical if col in sample_df.columns] + \
                              [col for col in platform_cols_onehot if col in platform_encoded.columns]

    processed_metadata = pd.concat([sample_df[meta_cols_numerical], platform_encoded[platform_cols_onehot]], axis=1)

    return processed_metadata[final_metadata_cols].values # Return only expected columns in correct order

# --- 4. Model Loading and Management ---
# Cache models to avoid reloading on each Streamlit rerun
@st.cache_resource(hash_funcs={BertTokenizer: id, BertModel: id})
def load_model(model_name, device, num_metadata_features=None, total_num_nodes=None, image_feature_dims=None, num_event_types=3):
    model = None
    # Fix: Ensure model path matches saved filename
    model_path_base = model_name.lower().replace(" ", "_")
    if model_name == "BiLSTM-Att":
        model_path_base = "bilstm_attention"
    elif model_name == "CTPP-GNN": # Specific handling for CTPP-GNN filename
        model_path_base = "ctpp_gnn"

    model_path = os.path.join(output_path, f'best_{model_path_base}_model.pth')
    st.write(f"Attempting to load {model_name} from {model_path}...")

    if model_name == "BiLSTM-Att":
        vocab_size = tokenizer.vocab_size
        embedding_dim = 300
        hidden_dim = 256
        num_layers = 2
        output_dim = 1
        model = BiLSTMAttentionClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, num_metadata_features, output_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name == "GCN":
        input_dim = 75 # Combined CNN-BERT (64) + metadata (9) + platform (2)
        hidden_dim = 64
        output_dim = 1
        model = GCNClassifier(input_dim, hidden_dim, output_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name == "TGN":
        input_dim_tgn = 75 # Combined CNN-BERT (64) + metadata (9) + platform (2)
        node_feat_dim = 64
        message_dim = 64
        memory_dim = 64
        hidden_dim_tgn = 64
        output_dim_tgn = 1
        model = TemporalGraphNetwork(input_dim_tgn, node_feat_dim, message_dim, memory_dim, hidden_dim_tgn, total_num_nodes, output_dim_tgn).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name == "MMT":
        vocab_size = tokenizer.vocab_size
        text_embedding_dim = 128
        text_hidden_dim = 128
        text_num_heads = 4
        text_num_layers = 2
        img_input_dim = image_feature_dims[1] # feature_dim_2
        img_hidden_dim = 128
        img_num_heads = 4
        img_num_layers = 2
        classifier_hidden_dim = 128
        output_dim = 1
        model = MultimodalTransformer(
            vocab_size, text_embedding_dim, text_hidden_dim, text_num_heads, text_num_layers,
            img_input_dim, img_hidden_dim, img_num_heads, img_num_layers,
            num_metadata_features, classifier_hidden_dim, output_dim
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name == "CTPP-GNN":
        input_dim_ctpp = 75
        node_embedding_dim = 64
        gcn_hidden_dim = 64
        tpp_hidden_dim = 64
        aggregator_hidden_dim = 64
        output_dim_ctpp = 1
        gcn_num_layers = 1
        dropout_rate_ctpp = 0.3
        model = CTPPGNN(
            input_dim=input_dim_ctpp, num_event_types=num_event_types, node_embedding_dim=node_embedding_dim,
            gcn_hidden_dim=gcn_hidden_dim, tpp_hidden_dim=tpp_hidden_dim, aggregator_hidden_dim=aggregator_hidden_dim,
            output_dim=output_dim_ctpp, gcn_num_layers=gcn_num_layers, dropout_rate=dropout_rate_ctpp
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(output_path, 'xgb_model_tuned.json')) # Load the tuned model
    else:
        st.error(f"Unknown model: {model_name}")
        return None

    if model_name != "XGBoost":
        model.eval() # Set PyTorch models to evaluation mode
    st.success(f"{model_name} loaded successfully!")
    return model

# Load CNN model (used by GCN, TGN, MMT, CTPP-GNN to process BERT embeddings)
@st.cache_resource
def load_cnn_model(device):
    cnn_model = CNN().to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(output_path, 'cnn_model.pth'), map_location=device))
    cnn_model.eval()
    return cnn_model
cnn_model = load_cnn_model(device)


# --- 5. Inference Pipeline Functions ---
# Shared inference logic for PyTorch models
def predict_pytorch_model(model, inputs, model_type, num_metadata_features_bilstm_mmt, total_num_nodes_tgn, image_feature_dims, num_event_types_ctpp):
    outputs = None
    attention_weights = None # Initialize attention weights
    if model_type == "BiLSTM-Att":
        bio_text, metadata_dict = inputs # Changed metadata_np to metadata_dict
        encoded_inputs = tokenizer(bio_text, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        # Preprocess metadata using the correct metadata columns (meta_cols_numerical_inference_mmt)
        processed_metadata = preprocess_metadata(metadata_dict, meta_cols_numerical_inference_mmt, platform_cols_inference)
        metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs, attention_weights = model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], metadata_tensor) # CAPTURE ATTENTION WEIGHTS
    elif model_type == "GCN":
        bio_text, metadata_np = inputs
        # 1. Get BERT embeddings
        bert_embeddings = get_bert_embeddings_inference(bio_text, tokenizer, bert_model, device)
        # 2. Get CNN features from BERT embeddings
        cnn_features = get_cnn_features_inference(bert_embeddings, cnn_model, device)
        # 3. Preprocess metadata (using meta_cols_numerical_inference for GCN)
        processed_metadata = preprocess_metadata(metadata_np, meta_cols_numerical_inference, platform_cols_inference)
        # 4. Combine CNN features and metadata
        combined_features = np.hstack([cnn_features, processed_metadata])
        features_tensor = torch.tensor(combined_features, dtype=torch.float32).to(device)
        # 5. Create sparse identity adjacency matrix
        adj = create_sparse_identity_adj(features_tensor.shape[0], device)
        with torch.no_grad():
            outputs = model(features_tensor, adj)
    elif model_type == "TGN":
        bio_text, metadata_np = inputs
        # 1. Get BERT embeddings
        bert_embeddings = get_bert_embeddings_inference(bio_text, tokenizer, bert_model, device)
        # 2. Get CNN features from BERT embeddings
        cnn_features = get_cnn_features_inference(bert_embeddings, cnn_model, device)
        # 3. Preprocess metadata (using meta_cols_numerical_inference for TGN)
        processed_metadata = preprocess_metadata(metadata_np, meta_cols_numerical_inference, platform_cols_inference)
        # 4. Combine CNN features and metadata
        combined_features = np.hstack([cnn_features, processed_metadata])
        features_tensor = torch.tensor(combined_features, dtype=torch.float32).to(device) # Add batch dim

        # Create synthetic temporal context for a single event
        src_nodes = torch.tensor([0], dtype=torch.long).to(device)
        dst_nodes = torch.tensor([0], dtype=torch.long).to(device)
        t_events = torch.tensor([0.0], dtype=torch.float32).to(device)
        messages = features_tensor # Node features serve as messages

        with torch.no_grad():
            outputs = model(features_tensor, src_nodes, dst_nodes, t_events, messages)
    elif model_type == "MMT":
        bio_text, metadata_np, img_features_np = inputs
        # Text features
        # Use max_length=64 for MMT as defined in mmt_collate_batch
        encoded_inputs = tokenizer(bio_text, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        # Metadata features (using meta_cols_numerical_inference_mmt for MMT)
        processed_metadata = preprocess_metadata(metadata_np, meta_cols_numerical_inference_mmt, platform_cols_inference)
        metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
        # Image features
        img_features_tensor = torch.tensor(img_features_np, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim

        with torch.no_grad():
            outputs = model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], metadata_tensor, img_features_tensor)
    elif model_type == "CTPP-GNN":
        bio_text, metadata_np = inputs
        # 1. Get BERT embeddings
        bert_embeddings = get_bert_embeddings_inference(bio_text, tokenizer, bert_model, device)
        # 2. Get CNN features from BERT embeddings
        cnn_features = get_cnn_features_inference(bert_embeddings, cnn_model, device)
        # 3. Preprocess metadata (using meta_cols_numerical_inference for CTPP-GNN)
        processed_metadata = preprocess_metadata(metadata_np, meta_cols_numerical_inference, platform_cols_inference)
        # 4. Combine CNN features and metadata
        combined_features = np.hstack([cnn_features, processed_metadata])
        event_features_tensor = torch.tensor(combined_features, dtype=torch.float32).to(device) # Removed .unsqueeze(0)

        # Synthetic temporal context for a single event
        timestamps = torch.tensor([0.0], dtype=torch.float32).to(device)
        src_nodes = torch.tensor([0], dtype=torch.long).to(device)
        dst_nodes = torch.tensor([0], dtype=torch.long).to(device)
        edge_types = torch.tensor([0], dtype=torch.long).to(device) # Arbitrary edge type
        edge_index = create_sparse_identity_adj(event_features_tensor.shape[0], device)

        with torch.no_grad():
            outputs = model(event_features_tensor, timestamps, src_nodes, dst_nodes, edge_types, edge_index)

    if outputs is not None:
        outputs = outputs.squeeze()
        probabilities = torch.sigmoid(outputs).item()
        prediction = 1 if probabilities >= 0.5 else 0
        return prediction, probabilities, attention_weights # RETURN ATTENTION WEIGHTS
    return -1, 0.5, None # Default if no output

def predict_xgboost_model(model, inputs):
    bio_text, metadata_np = inputs
    # 1. Get BERT embeddings
    bert_embeddings = get_bert_embeddings_inference(bio_text, tokenizer, bert_model, device)
    # 2. Get CNN features from BERT embeddings
    cnn_features = get_cnn_features_inference(bert_embeddings, cnn_model, device)
    # 3. Preprocess metadata (using meta_cols_numerical_inference for XGBoost)
    processed_metadata = preprocess_metadata(metadata_np, meta_cols_numerical_inference, platform_cols_inference)
    # 4. Combine CNN features and metadata
    combined_features = np.hstack([cnn_features, processed_metadata])

    # XGBoost expects numpy array
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0][1] # Probability of class 1
    return prediction, probabilities, None # No attention weights for XGBoost

# --- Reddit API Integration ---
# PRAW setup (replace with your actual credentials or env vars)
# Fixed: Remove default hardcoded values, rely solely on os.environ
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "fake_account_detector/1.0 by Winter_Lingonberry60"

# PRAW instance, cached to prevent re-creation
@st.cache_resource
def get_reddit_api(client_id, client_secret, user_agent):
    if not client_id or not client_secret:
        st.warning("Reddit API credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET) not found in environment variables. Please configure them.")
        return None
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Test authentication
        _ = reddit.user.me()
        return reddit
    except Exception as e:
        st.warning(f"Could not connect to Reddit API. Please check credentials. Error: {e}")
        return None

def fetch_reddit_data(username, reddit_api_instance, post_limit=100, comment_limit=100):
    posts_data = []
    comments_data = []
    all_timestamps = []
    user_external_url = 0 # Default to 0

    if not reddit_api_instance:
        st.error("Reddit API is not initialized. Cannot fetch data.")
        return None # Indicate API not available

    try:
        user = reddit_api_instance.redditor(username)
        # Check if user exists by trying to access an attribute that requires user data
        try:
            user_creation_date = user.created_utc # Access any attribute that would fail if user not found
        except Exception as e:
            if "404" in str(e): # PRAW raises 404 for non-existent users
                st.warning(f"Reddit user '{username}' not found.")
            else:
                st.error(f"Error accessing Reddit user '{username}' details: {e}")
            return None # Indicate user not found or error

        # Fetch external URL
        if hasattr(user, 'external_url') and user.external_url:
            user_external_url = 1
        else:
            user_external_url = 0

        # Fetch posts
        for submission in user.submissions.new(limit=post_limit):
            posts_data.append({
                'text': submission.title + " " + submission.selftext,
                'created_utc': submission.created_utc,
                'score': submission.score,
                'num_comments': submission.num_comments
            })
            all_timestamps.append(submission.created_utc)

        # Fetch comments
        for comment in user.comments.new(limit=comment_limit):
            comments_data.append({
                'text': comment.body,
                'created_utc': comment.created_utc,
                'score': comment.score
            })
            all_timestamps.append(comment.created_utc)

        return {"posts": posts_data, "comments": comments_data, "all_timestamps": all_timestamps, "user_external_url": user_external_url}

    except Exception as e:
        st.error(f"Error fetching Reddit data for user '{username}': {e}. Please ensure the username is correct and check API access if it's a persistent issue.")
        return None


# --- Streamlit UI ---

st.title("🤖 Fake Information Detection Across Internet")
st.markdown("---")

st.write("""
This application uses advanced machine learning models (BiLSTM with Attention, GCN, TGN, MMT, CTPP-GNN, and XGBoost)
to classify social media profiles as 'Real/Human' or 'Fake/Automated' based on textual bio, numerical metadata,
and in some cases, synthetic image features or temporal event patterns.
""")

# How-to section
with st.expander("❓ How to use this app"):
    st.markdown("""
    1.  **Choose a classification method**: You can either input text and metadata directly, or fetch data from a Reddit profile.
    2.  **Input data**:
        *   **Text & Metadata Input**: Paste a user's bio text and fill in the corresponding numerical features (followers, posts, etc.).
        *   **Reddit Profile Input**: Provide a Reddit username. The app will attempt to fetch posts and comments using the Reddit API (PRAW). *Note: Reddit API credentials must be configured as environment variables (e.g., in Colab's secrets panel) for this to work.* Set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` as environment variables.
    3.  **Select a Model**: Choose one of the available models from the dropdown.
    4.  **Click 'Classify'**: The app will process your input and display the prediction (Real/Human or Fake/Automated) along with a confidence score.
    """)

st.markdown("---")

# Model selection
model_options = ["XGBoost", "BiLSTM-Att", "GCN", "TGN", "MMT", "CTPP-GNN"]
selected_model = st.selectbox("Select a Classification Model:", model_options)

# Load relevant common parameters for models
@st.cache_resource
def get_common_model_params():
    # Placeholder for actual values, ensure they match training
    # Dynamically infer num_metadata_features without relying on external CSV files

    # Fix 2: Metadata columns for inference (must match what was used in training for GCN/TGN/XGBoost/CTPP-GNN)
    # These are the 9 columns that combined with CNN (64) + platform (2) = 75 total
    meta_cols_numerical_inference = ['followers_count', 'following_count', 'post_count', 'username_length',
                                     'username_digit_count', 'mean_likes', 'mean_comments', 'mean_hashtags',
                                     'upload_interval_std']

    # These are the 12 columns for MMT and BiLSTM-Att
    meta_cols_numerical_inference_mmt = ['followers_count', 'following_count', 'post_count', 'username_length',
                                         'username_digit_count', 'mean_likes', 'mean_comments', 'mean_hashtags',
                                         'upload_interval_std', 'userHasHighlighReels', 'userHasExternalUrl', 'userTagsCount']

    platform_cols_inference = ['platform_instagram', 'platform_twitter']

    # Dynamically determine num_metadata_features for MMT/BiLSTM-Att
    sample_metadata_dict_for_count = {col: 0 for col in meta_cols_numerical_inference_mmt} # Use MMT's meta cols
    sample_metadata_dict_for_count['platform'] = 'twitter' # Example platform

    sample_df_for_count = pd.DataFrame([sample_metadata_dict_for_count])
    platform_encoded_for_count = pd.get_dummies(sample_df_for_count['platform'], prefix='platform').astype(int)

    for p_col in platform_cols_inference:
        if p_col not in platform_encoded_for_count.columns:
            platform_encoded_for_count[p_col] = 0

    combined_metadata_for_count = pd.concat([sample_df_for_count[meta_cols_numerical_inference_mmt], platform_encoded_for_count[platform_cols_inference]], axis=1)
    num_metadata_features_bilstm_mmt = len(combined_metadata_for_count.columns) # This will be 12 + 2 = 14

    total_num_nodes_tgn = 10000 # Matches training data size
    image_feature_dims = (8, 128) # Matches synthetic image feature generation
    num_event_types_ctpp = 3 # Matches synthetic event generation

    return num_metadata_features_bilstm_mmt, total_num_nodes_tgn, image_feature_dims, num_event_types_ctpp, meta_cols_numerical_inference, meta_cols_numerical_inference_mmt, platform_cols_inference

num_metadata_features_bilstm_mmt, total_num_nodes_tgn, image_feature_dims, num_event_types_ctpp, meta_cols_numerical_inference, meta_cols_numerical_inference_mmt, platform_cols_inference = get_common_model_params()

# Load the selected model
# Fix 3: Pass num_metadata_features_bilstm_mmt for BiLSTM-Att and MMT
current_model = load_model(selected_model, device, num_metadata_features_bilstm_mmt, total_num_nodes_tgn, image_feature_dims, num_event_types_ctpp)

# --- Input Sections ---
input_method = st.radio("Choose Input Method:", ("Manual Text & Metadata Input", "Reddit Profile Analysis"))

bio_text = ""
metadata_input = {}

if input_method == "Manual Text & Metadata Input":
    st.subheader("Manual Profile Input")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bio Text**")
        bio_text = st.text_area("Enter the user's bio text:", value=st.session_state.get('manual_bio_text', "I love machine learning and data science. Sharing insights and tutorials."), height=150, help="This is the descriptive text about the user or account.")
        st.session_state.manual_bio_text = bio_text # Store current value

    with col2:
        st.markdown("**Numerical Metadata**")
        # Initialize default values
        default_metadata = {
            'followers_count': 1000,
            'following_count': 500,
            'post_count': 100,
            'username_length': 10,
            'username_digit_count': 0,
            'mean_likes': 50,
            'mean_comments': 5,
            'mean_hashtags': 3,
            'upload_interval_std': 86400.0, # 1 day in seconds
            'userHasHighlighReels': 0, # Binary 0/1
            'userHasExternalUrl': 0,   # Binary 0/1
            'userTagsCount': 0,
            'platform': 'twitter'
        }

        # Use session state for persistent input values
        for key, default_val in default_metadata.items():
            if key not in st.session_state:
                st.session_state[key] = default_val

        # Collect inputs, update session state on change
        metadata_input['followers_count'] = st.number_input("Followers Count", min_value=0, value=st.session_state.followers_count, key='followers_count_input')
        st.session_state.followers_count = metadata_input['followers_count']
        metadata_input['following_count'] = st.number_input("Following Count", min_value=0, value=st.session_state.following_count, key='following_count_input')
        st.session_state.following_count = metadata_input['following_count']
        metadata_input['post_count'] = st.number_input("Post Count", min_value=0, value=st.session_state.post_count, key='post_count_input')
        st.session_state.post_count = metadata_input['post_count']
        metadata_input['username_length'] = st.number_input("Username Length", min_value=0, value=st.session_state.username_length, key='username_length_input')
        st.session_state.username_length = metadata_input['username_length']
        metadata_input['username_digit_count'] = st.number_input("Username Digit Count", min_value=0, value=st.session_state.username_digit_count, key='username_digit_count_input')
        st.session_state.username_digit_count = metadata_input['username_digit_count']
        metadata_input['mean_likes'] = st.number_input("Mean Likes (if applicable)", min_value=0, value=st.session_state.mean_likes, key='mean_likes_input')
        st.session_state.mean_likes = metadata_input['mean_likes']
        metadata_input['mean_comments'] = st.number_input("Mean Comments (if applicable)", min_value=0, value=st.session_state.mean_comments, key='mean_comments_input')
        st.session_state.mean_comments = metadata_input['mean_comments']
        metadata_input['mean_hashtags'] = st.number_input("Mean Hashtags (if applicable)", min_value=0, value=st.session_state.mean_hashtags, key='mean_hashtags_input')
        st.session_state.mean_hashtags = metadata_input['mean_hashtags']
        metadata_input['upload_interval_std'] = st.number_input("Upload Interval Std (seconds)", min_value=0.0, value=st.session_state.upload_interval_std, format="%.2f", key='upload_interval_std_input')
        st.session_state.upload_interval_std = metadata_input['upload_interval_std']
        metadata_input['userHasHighlighReels'] = st.checkbox("Has Highlight Reels?", value=bool(st.session_state.userHasHighlighReels), key='userHasHighlighReels_input')
        st.session_state.userHasHighlighReels = int(metadata_input['userHasHighlighReels'])
        metadata_input['userHasExternalUrl'] = st.checkbox("Has External URL?", value=bool(st.session_state.userHasExternalUrl), key='userHasExternalUrl_input')
        st.session_state.userHasExternalUrl = int(metadata_input['userHasExternalUrl'])
        metadata_input['userTagsCount'] = st.number_input("User Tags Count", min_value=0, value=st.session_state.userTagsCount, key='userTagsCount_input')
        st.session_state.userTagsCount = metadata_input['userTagsCount']
        metadata_input['platform'] = st.selectbox("Platform", ['twitter', 'instagram'], index=['twitter', 'instagram'].index(st.session_state.platform), key='platform_input')
        st.session_state.platform = metadata_input['platform']

elif input_method == "Reddit Profile Analysis":
    st.subheader("Reddit Profile Analysis")
    reddit_username = st.text_input("Enter Reddit Username:", value=st.session_state.get('reddit_username', "spez"))
    st.session_state.reddit_username = reddit_username

    post_limit = st.number_input("Post Limit", min_value=1, value=st.session_state.get('reddit_post_limit', 100))
    st.session_state.reddit_post_limit = post_limit
    comment_limit = st.number_input("Comment Limit", min_value=1, value=st.session_state.get('reddit_comment_limit', 100))
    st.session_state.reddit_comment_limit = comment_limit

    reddit_api = get_reddit_api(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
    if not reddit_api:
        # Updated warning message here
        st.warning("Reddit API credentials not configured. Please add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` to environment variables (e.g., Colab secrets panel).")
        st.stop()

    if st.button("Fetch Reddit Data"):
        with st.spinner(f"Fetching {post_limit} posts and {comment_limit} comments for u/{reddit_username}... This may take a moment."):
            fetched_data = fetch_reddit_data(reddit_username, reddit_api, post_limit, comment_limit)

            if fetched_data is None: # Error occurred or user not found in fetching
                st.warning("Failed to fetch Reddit data. Please check logs for details or try a different username.")
                # Clear session state data if fetch failed
                if 'reddit_bio_text' in st.session_state: del st.session_state.reddit_bio_text
                if 'reddit_metadata_input' in st.session_state: del st.session_state.reddit_metadata_input
            else:
                posts_data = fetched_data["posts"]
                comments_data = fetched_data["comments"]
                all_timestamps = fetched_data["all_timestamps"]
                user_external_url = fetched_data["user_external_url"]

                if not posts_data and not comments_data:
                    st.warning(f"No posts or comments found for user '{reddit_username}' within the specified limits.")
                    # Clear session state data if no data found
                    if 'reddit_bio_text' in st.session_state: del st.session_state.reddit_bio_text
                    if 'reddit_metadata_input' in st.session_state: del st.session_state.reddit_metadata_input
                else:
                    # Aggregate bio text from posts and comments
                    all_text = [p['text'] for p in posts_data] + [c['text'] for c in comments_data]
                    bio_text = " ".join(all_text)[:2000] # Truncate to avoid excessive length

                    # Calculate upload_interval_std
                    upload_interval_std = 0.0
                    if len(all_timestamps) >= 2:
                        sorted_timestamps = sorted(all_timestamps)
                        time_diffs = np.diff(sorted_timestamps)
                        if len(time_diffs) > 0:
                           upload_interval_std = np.std(time_diffs)

                    # Aggregate numerical metadata
                    num_posts_total = len(posts_data)
                    num_comments_total = len(comments_data)

                    metadata_input = {
                        'followers_count': 0, # Reddit doesn't expose followers count like Insta/Twitter
                        'following_count': 0, # Similarly, following count
                        'post_count': num_posts_total,
                        'username_length': len(reddit_username),
                        'username_digit_count': sum(c.isdigit() for c in reddit_username),
                        'mean_likes': np.mean([p['score'] for p in posts_data if 'score' in p]) if num_posts_total > 0 else 0,
                        'mean_comments': np.mean([p['num_comments'] for p in posts_data if 'num_comments' in p]) if num_posts_total > 0 else 0,
                        'mean_hashtags': 0, # Reddit doesn't use hashtags in the same way
                        'upload_interval_std': upload_interval_std,
                        'userHasHighlighReels': 0,
                        'userHasExternalUrl': user_external_url,
                        'userTagsCount': 0,
                        'platform': 'reddit'
                    }

                    st.success(f"Successfully fetched {num_posts_total} posts and {num_comments_total} comments for u/{reddit_username}.")

                    st.markdown("**Aggregated Bio Text:**")
                    st.text_area("Bio Text (first 2000 chars):", bio_text, height=150, key="reddit_bio_text_display")

                    st.markdown("**Activity Timeline:**")
                    if all_timestamps:
                        earliest_ts = datetime.fromtimestamp(min(all_timestamps))
                        latest_ts = datetime.fromtimestamp(max(all_timestamps))
                        st.write(f"Earliest activity: {earliest_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"Latest activity: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.write("No activities found for timeline summary.")

                    st.markdown("**Numerical Metadata Summary:**")
                    st.write(f"Username Length: {metadata_input['username_length']}, Digits: {metadata_input['username_digit_count']}")
                    st.write(f"Posts: {metadata_input['post_count']}, Comments: {num_comments_total}")
                    st.write(f"Mean Likes (Posts): {metadata_input['mean_likes']:.2f}, Mean Comments (Posts): {metadata_input['mean_comments']:.2f}")
                    st.write(f"Upload Interval Std (seconds): {metadata_input['upload_interval_std']:.2f}")
                    st.write(f"Has External URL: {'Yes' if metadata_input['userHasExternalUrl'] else 'No'}")

                    st.session_state.reddit_bio_text = bio_text
                    st.session_state.reddit_metadata_input = metadata_input

    # Use data from session state if already fetched (for classification without re-fetching)
    if 'reddit_bio_text' in st.session_state and 'reddit_metadata_input' in st.session_state:
        bio_text = st.session_state.reddit_bio_text
        metadata_input = st.session_state.reddit_metadata_input
        # Display these in a collapsed expander to keep UI clean
        with st.expander("Currently Loaded Reddit Data"):
            st.text_area("Bio Text:", bio_text, height=100, disabled=True)
            st.json(metadata_input)

# --- Classification Button ---
if st.button("Classify Profile", key="classify_button"):
    if not current_model:
        st.error("Please select and load a model first.")
        st.stop()

    if not bio_text:
        st.warning("Please provide bio text for classification.")
        st.stop()

    if not metadata_input:
        st.warning("Please provide metadata for classification.")
        st.stop()

    # Special handling for MMT to generate synthetic image features if not provided (for manual input)
    img_features_for_mmt = None
    if selected_model == "MMT":
        # For MMT, we need synthetic image features for inference in this demo
        # In a real app, these would come from an image encoder
        img_features_for_mmt = np.random.rand(image_feature_dims[0], image_feature_dims[1]).astype(np.float32)

    st.info(f"Classifying with {selected_model}...")

    try:
        prediction, probability, attention_weights = (0, 0.5, None) # Initialize
        if selected_model == "XGBoost":
            prediction, probability, _ = predict_xgboost_model(current_model, (bio_text, metadata_input))
        else:
            inputs = (bio_text, metadata_input)
            if selected_model == "MMT":
                inputs = (bio_text, metadata_input, img_features_for_mmt)
            # Use correct metadata column list based on model type
            _meta_cols_numerical_for_model = meta_cols_numerical_inference_mmt if selected_model in ["BiLSTM-Att", "MMT"] else meta_cols_numerical_inference
            prediction, probability, attention_weights = predict_pytorch_model(current_model, inputs, selected_model,
                                                            num_metadata_features_bilstm_mmt, total_num_nodes_tgn,
                                                            image_feature_dims, num_event_types_ctpp)

        label_map = {0: "Real/Human 👤", 1: "Fake/Automated 🤖"}
        predicted_label = label_map[prediction]
        confidence = f"{probability:.2f}"

        st.subheader("Classification Result:")
        st.metric("Predicted Label", predicted_label, delta=None)
        st.metric("Confidence Score", confidence, delta=None)

        st.markdown("---")
        st.write("### Explanation:")
        explanation_text = ""
        if prediction == 1:
            explanation_text = f"The model has classified this profile as **Fake/Automated** with a confidence of {confidence}. This suggests that the profile's characteristics (text, metadata, and potentially other modalities depending on the model) align more closely with patterns observed in automated or fake accounts."
        else:
            explanation_text = f"The model has classified this profile as **Real/Human** with a confidence of {confidence}. This indicates that the profile's characteristics are more consistent with those of genuine human users."
        st.write(explanation_text)

        st.write("""
### What these labels mean:
- **Real/Human (0)**: Accounts exhibiting behavior and features typical of genuine human users.
- **Fake/Automated (1)**: Accounts exhibiting characteristics often associated with bots, automated scripts, or intentionally deceptive profiles.
""")

        if selected_model == "BiLSTM-Att" and attention_weights is not None:
            st.markdown("### Highlighted Bio Text (Attention Scores):")
            # Decode tokens first
            tokenized_input = tokenizer(bio_text, truncation=True, max_length=128, return_tensors='pt')
            tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])

            # Ensure attention weights length matches tokens length considering padding
            # We use attention_mask to correctly extract relevant weights
            effective_length = tokenized_input['attention_mask'][0].sum().item()
            attention_weights_cpu = attention_weights[0, :effective_length].cpu().numpy()
            tokens_effective = tokens[:effective_length]

            # Remove special tokens (CLS, SEP) and their weights if they exist in the effective range
            display_tokens = []
            display_weights = []
            for i, token in enumerate(tokens_effective):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    display_tokens.append(token)
                    display_weights.append(attention_weights_cpu[i])

            if display_weights:
                # Normalize weights for coloring, handling case where all weights are the same
                min_attn, max_attn = np.min(display_weights), np.max(display_weights)
                if max_attn == min_attn:
                    normalized_weights = np.zeros_like(display_weights)
                else:
                    normalized_weights = (display_weights - min_attn) / (max_attn - min_attn)

                highlighted_text = ""
                for token, weight in zip(display_tokens, normalized_weights):
                    # Use a color scale (e.g., from light red to dark red) for attention
                    color_intensity = int(255 * weight) # Red channel intensity
                    highlighted_text += f"<span style='background-color: rgba(255, 0, 0, {weight:.2f}); padding: 2px;'>{token.replace('##', '')}</span> " # Remove ## for readability
                st.markdown(highlighted_text, unsafe_allow_html=True)
                st.caption("Tokens highlighted in red indicate higher attention given by the model.")
            else:
                st.write("No meaningful tokens to display attention for.")

        # --- JSON Report Generation ---
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_used": selected_model,
            "input_method": input_method,
            "bio_text_input": bio_text,
            "metadata_input": metadata_input,
            "prediction": int(prediction),
            "predicted_label": label_map[prediction].replace(' 👤', '').replace(' 🤖', ''),
            "confidence_score": probability,
            "explanation": explanation_text,
            "attention_weights": attention_weights.tolist() if attention_weights is not None else None
        }

        json_report = json.dumps(report_data, indent=4)
        st.download_button(
            label="Download Classification Report (JSON)",
            data=json_report,
            file_name=f"bot_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        st.exception(e)

st.markdown("---")
st.write("Developed as part of a Capstone Project. Leveraging advanced ML/DL models for social media bot detection.")

# Footer with GPU usage (optional, for debugging/monitoring)
if torch.cuda.is_available():
    st.sidebar.subheader("GPU Usage")
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    st.sidebar.write(f"Allocated: {allocated:.2f} MB")
    st.sidebar.write(f"Reserved: {reserved:.2f} MB")
