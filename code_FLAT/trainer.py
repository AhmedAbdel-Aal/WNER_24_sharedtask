from torch.optim import AdamW

criterion_main = nn.CrossEntropyLoss()
criterion_subtype = nn.BCEWithLogitsLoss()

optimizer = AdamW(model.parameters(), lr=2e-5)

def train(model, dataloader, optimizer, criterion_main, criterion_subtype, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            main_labels = batch['labels']
            subtype_labels = batch['sub_tags']
            
            optimizer.zero_grad()
            
            # Forward pass
            main_logits, subtype_logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss_main = criterion_main(main_logits.transpose(1, 2), main_labels)
            loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
            total_loss = loss_main + loss_subtype
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            total_loss += total_loss.item()

        print(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

# Assuming 'dataloader' is defined and loaded with the appropriate batched data
# train(model, dataloader, optimizer, criterion_main, criterion_subtype)

def evaluate(model, dataloader, criterion_main, criterion_subtype):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            main_labels = batch['main_labels']
            subtype_labels = batch['subtype_labels']
            
            # Forward pass
            main_logits, subtype_logits = model(input_ids, attention_mask)
