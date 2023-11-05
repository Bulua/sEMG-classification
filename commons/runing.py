import sys
import torch

from tqdm import tqdm


def train(model, 
          train_dataloader,
          valid_dataloader,
          loss_func,
          epochs,
          optimizer,
          lr_scheduler,
          device):
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer,
                                                loss_func=loss_func,
                                                epoch=epoch,
                                                device=device)
        lr_scheduler.step()

        valid_loss, valid_acc = evaluate(model=model,
                                         dataloader=valid_dataloader,
                                         loss_func=loss_func,
                                         epoch=epoch,
                                         device=device)

        # print(train_loss, train_acc, valid_loss, valid_acc)

@torch.no_grad()
def evaluate(model, 
             dataloader,
             loss_func,
             epoch,
             device):

    model.eval()
    acc_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)

    sample_num = 0
    dataloader = tqdm(dataloader, file=sys.stdout)

    for step, data in enumerate(dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        _, idxs_pred = torch.max(pred, dim=1)
        # _, idxs_true = torch.max(labels, dim=1)
        acc_num += torch.eq(idxs_pred, labels).sum()

        loss = loss_func(pred, labels)
        acc_loss += loss

        dataloader.desc = '[valid epoch {}] loss: {:.3f}, acc: {:.3f}'.format(epoch,
                                                                             acc_loss.item() / (step + 1),
                                                                             acc_num.item() / sample_num)
    return acc_loss.item() / (step + 1), acc_num.item() / sample_num


def train_one_epoch(model,
          dataloader,
          optimizer,
          loss_func,
          epoch,
          device):

    model.train()
    acc_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        _, idxs_pred = torch.max(pred, dim=1)
        # _, idxs_true = torch.max(labels, dim=1)
        acc_num += torch.eq(idxs_pred, labels).sum()

        loss = loss_func(pred, labels)
        loss.backward()

        acc_loss += loss.detach()
        dataloader.desc = '[train epoch {}] loss: {:.3f}, acc: {:.3f}'.format(epoch,
                                                                            acc_loss.item() / (step + 1),
                                                                            acc_num.item() / sample_num)
        optimizer.step()
        optimizer.zero_grad()
    return acc_loss.item() / (step + 1), acc_num.item() / sample_num


if __name__ == '__main__':
    batch_size = 32
    data_size = 100
    feature_size = 12
    # datas = torch.arange(batch_size * data_size * feature_size) \
    #         .reshape((batch_size, data_size, feature_size))
    # labels = 
    print()
