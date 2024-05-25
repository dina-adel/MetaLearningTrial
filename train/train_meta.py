import copy

import torch
from data import generate_x, sin_generator, sample_meta_data
from model import MetaModel
from matplotlib import pyplot as plt


def train_meta_model(epochs=100000, model_path='../checkpoints/meta/model.pt'):
    def inner_loop(model_copy, optim, task_data):
        for task in task_data:
            data, labels = task[0], task[1]
            optim.zero_grad()
            outputs = model_copy(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            print(f'Loss: {loss.item():.4f}')
        return model_copy, optim

    def outer_loop(model_copy, optim, meta_data):
        for task_data in meta_data:
            new_model, optim = inner_loop(model_copy, optim, task_data)
        return new_model, optim

    model = MetaModel()
    loss_fn = torch.nn.MSELoss()
    optim_train = torch.optim.SGD(model.parameters(), lr=5e-3)
    optim_test = torch.optim.SGD(model.parameters(), lr=5e-3)

    train_data, test_data = sample_meta_data(num_tasks=10, num_samples_per_task=1000,
                                             num_train_samples_per_task=800, num_test_samples_per_task=200)

    for epoch in range(epochs):
        print("Working on Training Data -> epoch {}".format(epoch + 1))
        # first train on the train dataset
        train_model, optim_train = outer_loop(copy.deepcopy(model), optim_train, train_data)

        print("Working on Testing Data -> epoch {}".format(epoch + 1))
        # then use the test/query data
        test_model, optim_test = outer_loop(train_model, optim_test, test_data)

        # update original model using the test_model
        model = copy.deepcopy(test_model)

        for param_original, param_test in zip(model.parameters(), test_model.parameters()):
            param_original.data = param_test.data

    # Save the updated model
    torch.save(model.state_dict(), model_path)
    return


if __name__ == '__main__':
    print("Meta Training Started...")
    train_meta_model()
