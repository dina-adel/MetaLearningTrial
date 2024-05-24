import torch
from data import generate_x, sin_generator
from model import MetaModel


def train_meta_model(epochs=10000, batch_size=10, finetune_path=None,
                     test_sin_gen=None, model_path='../checkpoints/basic/model.pt'):

    model = MetaModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    if finetune_path:
        model.load_state_dict(torch.load(finetune_path))

    for epoch in range(epochs):
        optim.zero_grad()

        sin_gen = test_sin_gen if test_sin_gen else sin_generator()

        batch = generate_x(sample_size=batch_size)
        pred = model(batch)
        gt = sin_gen(batch)

        loss = loss_fn(pred, gt)
        loss.backward()

        optim.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    # save model after training
    torch.save(model.state_dict(), model_path)
    return model


if __name__ == '__main__':
    print("Meta Training Started...")
