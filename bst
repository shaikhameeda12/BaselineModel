
session_train_data = session_train_df.sample(frac=0.1, random_state=42)[['prev_items', 'next_item','locale']]

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Extract the session data from the session test dataset
session_test_data = session_test_df[['locale', 'prev_items']]

# Extract the relevant columns from the product train data
product_train_df = product_train_df[['id', 'color', 'size', 'brand', 'model', 'material', 'author']]

# Convert categorical features to numerical features using label encoding
label_encoder = LabelEncoder()
product_train_df['color'] = label_encoder.fit_transform(product_train_df['color'])
product_train_df['size'] = label_encoder.fit_transform(product_train_df['size'])
product_train_df['brand'] = label_encoder.fit_transform(product_train_df['brand'])
product_train_df['model'] = label_encoder.fit_transform(product_train_df['model'])
product_train_df['material'] = label_encoder.fit_transform(product_train_df['material'])
product_train_df['author'] = label_encoder.fit_transform(product_train_df['author'])

# Convert categorical features to numerical features using one-hot encoding
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(product_train_df[['color', 'size', 'brand', 'model', 'material', 'author']])
encoded_feature_names = encoder.get_feature_names_out(['color', 'size', 'brand', 'model', 'material', 'author'])
encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)

# Merge the encoded features with the product train data
product_train_df = pd.concat([product_train_df, encoded_features_df], axis=1)
product_train_df = product_train_df.drop(['color', 'size', 'brand', 'model', 'material', 'author'], axis=1)

# Merge the session and product data
merged_data = pd.merge(session_train_data, product_train_df, left_on='next_item', right_on='id', how='inner')

# Tokenize the session data in the merged dataset
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_session(session):
    tokenized_session = tokenizer.encode(session, add_special_tokens=True, max_length=1024)
    product_ids = tokenizer.convert_ids_to_tokens(tokenized_session)
    return product_ids

merged_data['product_ids'] = merged_data['prev_items'].apply(tokenize_session)

# Split the merged dataset into training, validation, and test sets
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

print("Training Data:")
print(train_data.head())
print("\nValidation Data:")
print(val_data.head())
print("\nTest Data:")
print(test_data.head())



session_train_data = session_train_df.sample(frac=0.1, random_state=42)[['prev_items', 'next_item','locale']]

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Extract the session data from the session test dataset
session_test_data = session_test_df[['locale', 'prev_items']]

# Extract the relevant columns from the product train data
product_train_df = product_train_df[['id', 'color', 'size', 'brand', 'model', 'material', 'author']]

# Convert categorical features to numerical features using label encoding
label_encoder = LabelEncoder()
product_train_df['color'] = label_encoder.fit_transform(product_train_df['color'])
product_train_df['size'] = label_encoder.fit_transform(product_train_df['size'])
product_train_df['brand'] = label_encoder.fit_transform(product_train_df['brand'])
product_train_df['model'] = label_encoder.fit_transform(product_train_df['model'])
product_train_df['material'] = label_encoder.fit_transform(product_train_df['material'])
product_train_df['author'] = label_encoder.fit_transform(product_train_df['author'])

# Convert categorical features to numerical features using one-hot encoding
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(product_train_df[['color', 'size', 'brand', 'model', 'material', 'author']])
encoded_feature_names = encoder.get_feature_names_out(['color', 'size', 'brand', 'model', 'material', 'author'])
encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)

# Merge the encoded features with the product train data
product_train_df = pd.concat([product_train_df, encoded_features_df], axis=1)
product_train_df = product_train_df.drop(['color', 'size', 'brand', 'model', 'material', 'author'], axis=1)

# Merge the session and product data
merged_data = pd.merge(session_train_data, product_train_df, left_on='next_item', right_on='id', how='inner')

# Tokenize the session data in the merged dataset
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_session(session):
    tokenized_session = tokenizer.encode(session, add_special_tokens=True, max_length=1024)
    product_ids = tokenizer.convert_ids_to_tokens(tokenized_session)
    return product_ids

merged_data['product_ids'] = merged_data['prev_items'].apply(tokenize_session)

# Split the merged dataset into training, validation, and test sets
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

print("Training Data:")
print(train_data.head())
print("\nValidation Data:")
print(val_data.head())
print("\nTest Data:")
print(test_data.head())

import torch

# Convert the data to PyTorch tensors
train_inputs = torch.from_numpy(train_data['product_ids'].to_numpy())
train_labels = torch.from_numpy(train_data['id'].to_numpy())
val_inputs = torch.from_numpy(val_data['product_ids'].to_numpy())
val_labels = torch.from_numpy(val_data['id'].to_numpy())
test_inputs = torch.from_numpy(test_data['product_ids'].to_numpy())
test_labels = torch.from_numpy(test_data['id'].to_numpy())


from torch.utils.data import TensorDataset, DataLoader

# Create a PyTorch DataLoader
batch_size = 32
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_inputs, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


from transformers import BertForSequenceClassification, AdamW
import torch.nn.functional as F

# Define the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(product_train_df))

# Set the hyperparameters
learning_rate = 2e-5
epochs = 3

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = F.cross_entropy

# Train the model
for epoch in range(epochs):
    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs, labels=batch_labels)
        loss = loss_fn(outputs.logits, batch_labels)
        loss.backward()
        optimizer.step()


import numpy as np

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    mrr = 0
    for batch_inputs, batch_labels in test_loader:
        outputs = model(batch_inputs)
        logits = outputs.logits
        _, predicted = torch.topk(logits, k=10, dim=1)
        predicted = predicted.cpu().numpy()
        labels = batch_labels.cpu().numpy()
        for i in range(len(predicted)):
            rank = np.where(predicted[i] == labels[i])[0][0] + 1
            mrr += 1 / rank
    mrr /= len(test_data)
    print("Mean Reciprocal Rank (MRR):", mrr)


