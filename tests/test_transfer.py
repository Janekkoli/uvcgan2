import unittest

import torch

from uvcgan2.config import TransferConfig
from uvcgan2.train.transfer import transfer_parameters


class TestTransfer(unittest.TestCase):

    def test_transfer_initializes_average_generator(self):
        source = torch.nn.Linear(2, 2)
        generator = torch.nn.Linear(2, 2)
        average_generator = torch.nn.Linear(2, 2)
        untouched_average = torch.nn.Linear(2, 2)

        with torch.no_grad():
            source.weight.fill_(1)
            source.bias.fill_(2)
            untouched_average.weight.fill_(3)
            untouched_average.bias.fill_(4)

        model = type('Model', (), {
            'models': {
                'gen_ab': generator,
                'avg_gen_ab': average_generator,
                'avg_gen_ba': untouched_average,
            },
        })()
        base_model = type('Model', (), {
            'models': {'encoder': source},
        })()
        transfer_config = TransferConfig(
            base_model='base',
            transfer_map={'gen_ab': 'encoder'},
        )

        transfer_parameters(model, base_model, transfer_config)

        for (actual, expected) in zip(
            average_generator.parameters(), source.parameters()
        ):
            self.assertTrue(torch.equal(actual, expected))

        self.assertTrue(torch.equal(untouched_average.weight, torch.full((2, 2), 3)))
        self.assertTrue(torch.equal(untouched_average.bias, torch.full((2,), 4)))


if __name__ == '__main__':
    unittest.main()
