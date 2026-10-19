import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import tensorflow as tf

from pbo.src.continuous import continuous
from pbo.src.discrete import discrete
from pbo.src.par_envs import par_envs
from pbo.src.train import train

ROOT = Path(__file__).resolve().parents[1]


def parameters(name):
    with (ROOT / 'envs' / (name + '.json')).open() as stream:
        values = json.load(stream)
    values.update(n_gen=2, n_ind=4, n_cpu=2, n_avg=1)
    for key in ('mu_epochs', 'sg_epochs', 'cr_epochs', 'epochs'):
        if key in values:
            values[key] = 1
    return SimpleNamespace(**values)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def exercise_agent(self, agent, networks):
        observations = np.ones((4, agent.obs_dim), dtype=np.float64)
        actions = agent.get_actions(observations, 4)
        self.assertEqual(actions.shape, (4, agent.act_dim))
        self.assertTrue(np.isfinite(actions).all())
        weights = [v for net in networks for v in net.trainable_variables]
        before = [v.numpy().copy() for v in weights]
        agent.store_transition(observations, actions, actions,
                               np.array([-4., -1., -3., -2.]), 4)
        agent.compute_advantages()
        agent.train_networks()
        self.assertTrue(all(np.isfinite(v.numpy()).all() for v in weights))
        self.assertTrue(any(not np.array_equal(old, v.numpy())
                            for old, v in zip(before, weights)))

    def test_continuous_policies_train(self):
        for pdf, dimensions in [('es', 1), ('es', 2),
                                ('cma-diag', 2), ('cma-full', 2)]:
            with self.subTest(pdf=pdf, dimensions=dimensions):
                params = parameters('sphere_2d')
                params.pdf = pdf
                agent = continuous(params, dimensions, 2)
                self.exercise_agent(agent, [agent.net_mu, agent.net_sg,
                                            agent.net_cr])

    def test_discrete_policy_trains_and_samples_valid_categories(self):
        agent = discrete(parameters('sphere_2d_discrete'), [3, 5], 2)
        self.exercise_agent(agent, [agent.net])
        actions = agent.get_actions(np.ones((1, 2)), 100)
        self.assertTrue((actions >= 0).all())
        self.assertTrue((actions < np.array([3, 5])).all())
        np.testing.assert_array_equal(actions, actions.astype(int))

    def test_parallel_environment_reward_shape_and_cleanup(self):
        with tempfile.TemporaryDirectory() as output:
            env = par_envs('sphere_2d', 2, output, str(ROOT / 'envs'))
            try:
                self.assertEqual(env.observe(2).shape, (2, 2))
                rewards, actions = env.step(np.zeros((2, 2)), 2, 0)
                self.assertEqual(rewards.shape, (2,))
                np.testing.assert_allclose(rewards, [-12.5, -12.5])
                self.assertEqual(actions.shape, (2, 2))
            finally:
                env.close()
            self.assertTrue(all(not process.is_alive() for process in env.proc))

    def test_continuous_and_discrete_end_to_end(self):
        for name in ('sphere_2d', 'sphere_2d_discrete'):
            with self.subTest(environment=name), tempfile.TemporaryDirectory() as output:
                params = parameters(name)
                # Exercise a final batch smaller than n_cpu.
                params.n_ind = 3
                train(params, output, str(ROOT / 'envs'), 0)
                history = np.loadtxt(Path(output) / 'pbo_0')
                best = np.loadtxt(Path(output) / 'pbo_bst_0')
                self.assertEqual(history.shape, (6, 5))
                self.assertEqual(best.shape, (2, 5))
                self.assertTrue(np.isfinite(history).all())
                self.assertTrue(np.isfinite(best).all())
                self.assertLessEqual(best[-1, 2], best[0, 2])

    def test_cli_single_generation_writes_plot(self):
        with tempfile.TemporaryDirectory() as output:
            params = vars(parameters('sphere_2d'))
            params.update(n_gen=1, n_cpu=1)
            config = Path(output) / 'sphere_2d.json'
            config.write_text(json.dumps(params))
            env = os.environ.copy()
            # The CLI resolves custom environment modules next to the JSON.
            (Path(output) / 'sphere_2d.py').write_text(
                'from envs.sphere_2d import sphere_2d\n')
            result = subprocess.run(
                [sys.executable, '-c', 'from pbo.src.main import main; main()', str(config)],
                cwd=output, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(output) / 'pbo.png').is_file())


if __name__ == '__main__':
    unittest.main()
