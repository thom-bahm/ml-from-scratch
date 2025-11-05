import torch
from tqdm import tqdm

def test_model(model, test_loader, criterion, loss_logger, device):
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):   
            
            # Forward pass of model
            outputs = model.forward(data.to(device))# TO DO           
            
            # Calculate the accuracy of the model
            # You'll need to accumulate the accuracy over multiple steps
            # TO DO
            # Number of correctly predicted outputs - returns a tensor of shape (1,1)
            max_predicted_value, predicted_class = torch.max(outputs, 1)
            correct_predictions += (predicted_class == target.to(device)).sum().item()
            total_predictions += target.shape[0]

            
            # Calculate the loss
            loss = criterion(outputs, target.to(device))# TO DO
            loss_logger.append(loss.item())
            
        acc = (correct_predictions/total_predictions) * 100.0
        return loss_logger, acc
