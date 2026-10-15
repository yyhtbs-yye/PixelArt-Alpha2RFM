"""RFM contracts: direction, loss, ODE/CFG, source balance and resume cursor."""
import unittest
import torch
from diffusion.rectified_flow import flow_matching_loss, sample_rfm
from datasets.rfm_data import BalancedFlowSampler


class RectifiedFlowTests(unittest.TestCase):
    def test_path_target_time_and_gradient(self):
        clean = torch.randn(3, 4, 2, 2)
        noise = torch.randn_like(clean)
        time = torch.tensor([0., .37, 1.])
        offset = torch.tensor(.25, requires_grad=True)

        def oracle(x, embedding_time, condition, mask):
            self.assertTrue(torch.equal(x[0], noise[0]))
            self.assertTrue(torch.equal(x[-1], clean[-1]))
            torch.testing.assert_close(embedding_time, 999 * (1-time))
            torch.testing.assert_close(x[1], .63*noise[1]+.37*clean[1])
            return clean - noise + offset

        loss = flow_matching_loss(oracle, clean, noise, time, None, None).mean()
        torch.testing.assert_close(loss, torch.tensor(.25**2))
        loss.backward()
        torch.testing.assert_close(offset.grad, torch.tensor(.5))

    def test_cfg_and_solver_direction_all_four_channels(self):
        noise = torch.randn(2, 4, 2, 2)
        condition = torch.ones(2, 1, 1, 1)
        empty = torch.zeros(1, 1, 1, 1)
        mask = torch.ones(2, 1, dtype=torch.bool)

        def model(x, time, y, mask):
            return y.expand_as(x)

        for method in ('heun', 'dopri5'):
            result, report = sample_rfm(model, noise, condition, mask, empty, mask[:1],
                                        cfg_scale=2., method=method, steps=4)
            torch.testing.assert_close(result, noise + 2, atol=2e-5, rtol=2e-5)
            self.assertGreater(report['nfe'], 0)
            if method == 'heun':
                self.assertEqual(report['nfe'], 8)

    def test_unequal_sources_balance_coverage_and_resume(self):
        lengths = [50, 77]
        rank_batches = [list(BalancedFlowSampler(lengths, 8, 4, rank, 43, 0, 20)) for rank in range(4)]
        seen = [set(), set()]
        for step in range(20):
            for rank in range(4):
                batch = rank_batches[rank][step]
                self.assertEqual(sum(index < 50 for index in batch), 4)
                self.assertEqual(sum(index >= 50 for index in batch), 4)
                self.assertEqual(len(set(batch)), 8)
                seen[0].update(index for index in batch if index < 50)
                seen[1].update(index for index in batch if index >= 50)
            if step < 3:  # Full global half-batches before either source's padded tail.
                combined = sum([rank_batches[rank][step] for rank in range(4)], [])
                self.assertEqual(len(set(combined)), 32)
        self.assertEqual(seen, [set(range(50)), set(range(50, 127))])
        for rank in range(4):
            resumed = list(BalancedFlowSampler(lengths, 8, 4, rank, 43, 7, 20))
            self.assertEqual(resumed, rank_batches[rank][7:])


if __name__ == '__main__':
    unittest.main()
