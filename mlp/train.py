from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, loss_logger, device):
    # batch_idx indicates what batch we are on.
    # data is of shape torch.size([64,1,28,28]), as there are 64 individual data entries
    # each with 1 channel (MNIST images are grayscale), and each image is 28x28.
    # target is of shape torch.size([64]), and each data entry is the label of the image (a digit 0-9)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):        # Forward pass of model
        outputs = model.forward(data.to(device))# TO DO
        
        # Calculate loss
        loss = criterion(outputs, target)# TO DO
        
        # Zero gradients
        # TO DO
        optimizer.zero_grad()
        # Backprop loss
        # TO DO
        loss.backward()
        # Optimization Step
        # TO DO
        optimizer.step()

        loss_logger.append(loss.item())

    return model, optimizer, loss_logger
