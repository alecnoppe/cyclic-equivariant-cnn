import torch
import pandas as pd


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, validation_dataloader, test_dataloader, device):
        """
        Class for training torch models.

        Args:
            model: nn.Module; pyTorch model
            optimizer: torch.optim; optimizer to use
            loss_fn: Callable; loss function that takes pred_y, y as input
            train_dataloader: torch.DataLoader; training dataloader
            validation_dataloader: torch.DataLoader; validation dataloader
            test_dataloader: torch.DataLoader; test dataloader
            device: str; torch device ("cuda:<idx>" or "cpu")
        """
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def train(self, best_model_path="models/best-model.pt", loss_path="results/loss.csv", n_epochs=50):
        """
        Train self.model for n_epochs, store best validation loss and save the best model.

        Args:
            best_model_path: str; path to save the best model
            loss_path: str; path to save the loss
            n_epochs: int; number of epochs to train the model
        """
        # Train for n_epochs, store best validation loss and save model if save_model is True
        best_vloss = 1_000_000
        out = {"loss":[], "vloss":[]}
        for epoch in range(1, n_epochs+1):
            # Train the model and record (validation loss)
            loss, vloss = self.train_step()
            out["loss"].append(loss)
            out["vloss"].append(vloss)
            print(f"EPOCH-{epoch:04d} Training loss: {loss:.4f}\t\t Validation loss: {vloss:.4f}")
            # If validation loss is the lowest it has been
            # save the model
            if vloss < best_vloss:
                best_vloss = vloss
                torch.save(self.model, best_model_path)
        # Save the results
        out = pd.DataFrame(out)
        out.to_csv(loss_path)

    def train_step(self):
        """
        Train the model for one epoch.

        Returns:
            float; average training loss (per sample)
            float; average validation loss (per sample)
        """
        # Training
        self.model.train(True)
        running_loss = 0
        avg_loss = 0
        
        for i, data in enumerate(self.train_dataloader):
            # Every object in train_loader is the current state x and a future state y
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            # Set gradients to zero for new batch
            self.optimizer.zero_grad()
            # Get model prediction, and compute loss
            pred_y = self.model(x)
            loss = self.loss_fn(pred_y, y)
            running_loss += loss.item()
            # Update model parameters
            loss.backward()
            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Adjust weights of the model
            self.optimizer.step()
        # Compute average training loss
        total_samples = self.train_dataloader.dataset.__len__()
        avg_loss = running_loss / total_samples
        
        # Validation
        running_vloss = 0
        avg_vloss = 0
        self.model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(self.validation_dataloader):
                # Every object in validation_loader is the current state x and a future state y
                val_x, val_y = vdata
                
                val_x, val_y = val_x.to(self.device), val_y.to(self.device)

                # Get model prediction, and compute loss
                pred_y = self.model(val_x)
                vloss = self.loss_fn(pred_y, val_y)
                running_vloss += vloss.item()
        # Compute average validation loss
        total_validation_samples = self.validation_dataloader.dataset.__len__()
        avg_vloss = running_vloss / total_validation_samples

        return avg_loss, avg_vloss
    
    def test(self, model):
        """
        Test the model on the test set.

        Args:
            model: nn.Module; model to test

        Returns:    
            float; average test loss (per batch)
        """
        running_tloss = 0
        with torch.no_grad():
            for i, tdata in enumerate(self.test_dataloader):
                # Every object in validation_loader is the current state x and a future state y
                test_x, test_y = tdata
                
                test_x, test_y = test_x.to(self.device), test_y.to(self.device)

                # Get model prediction, and compute loss
                pred_y = model(test_x)
                tloss = self.loss_fn(pred_y, test_y)
                running_tloss += tloss.item()
        avg_tloss = running_tloss / i

        return avg_tloss
