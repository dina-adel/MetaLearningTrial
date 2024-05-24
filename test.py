from model import MetaModel
from data import sin_generator, generate_x
from matplotlib import pyplot as plt
from train.train_basic import train_basic_model
import argparse
import torch

parser = argparse.ArgumentParser(prog='Meta-Learning Trial', )
parser.add_argument('--model_path', default='./checkpoints/basic/model.pt',
                    help='Path to the model checkpoint')
parser.add_argument('--epochs', default=3, type=int, help='fine-tuning epochs')
parser.add_argument('--finetune', action='store_true', help='fine-tune model on test')
args = parser.parse_args()

if __name__ == '__main__':
    model = MetaModel()
    model.load_state_dict(torch.load(args.model_path))

    test_x = generate_x(sample_size=100)
    sin_gen = sin_generator()

    if args.finetune:
        model = train_basic_model(epochs=args.epochs, finetune_path=args.model_path,
                                  test_sin_gen=sin_gen, model_path='./checkpoints/basic/model_finetune.pt')

    # test
    model.eval()
    torch.no_grad()

    pred = model(test_x)
    gt = sin_gen(test_x)

    plt.plot(test_x, gt)
    plt.plot(test_x, pred.detach().numpy())
    plt.show()
