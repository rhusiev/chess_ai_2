import torch

from chess_ai.data_for_train_prepare import SAVE_DIR, dataloader, device
from chess_ai.model_initialize import loss_function, model, optimizer

def save_model(model, optimizer, epoch="latest", i="latest"):
    print("Saving model...", end=" ")
    torch.save(model.state_dict(), f"{SAVE_DIR}/model_parameters_800_1200_epoch{epoch}_batch{i}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{SAVE_DIR}/model_and_optimizer_800_1200_epoch{epoch}.pth')
    print("Done!")


epochs = 2
for epoch in range(epochs):
    epoch += 4
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, consts, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()

        outputs = model(inputs, consts)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch: {epoch + 1: >2} | Batch: {i: >5} | loss: {running_loss / 100:.3f}")
            running_loss = 0.0

        if i % 5000 == 0:
            save_model(model, optimizer, epoch + 1, i)

save_model(model, optimizer)
print("Finished Training")

save_model(model, optimizer)
