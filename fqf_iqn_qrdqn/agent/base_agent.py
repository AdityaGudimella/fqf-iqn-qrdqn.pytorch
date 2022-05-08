from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import tqdm

from fqf_iqn_qrdqn.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from fqf_iqn_qrdqn.utils import RunningMeanStats, LinearAnneaer, use_morl

from morl import api, core, memories, external_utils as extu, learners, models


class BaseAgent(ABC):
    def __init__(
        self,
        env,
        test_env,
        log_dir,
        num_steps=5 * (10**7),
        batch_size=32,
        memory_size=10**6,
        gamma=0.99,
        multi_step=1,
        update_interval=4,
        target_update_interval=10000,
        start_steps=50000,
        epsilon_train=0.01,
        epsilon_eval=0.001,
        epsilon_decay_steps=250000,
        double_q_learning=False,
        dueling_net=False,
        noisy_net=False,
        use_per=False,
        log_interval=100,
        eval_interval=250000,
        num_eval_steps=125000,
        max_episode_steps=27000,
        grad_cliping=5.0,
        cuda=True,
        seed=0,
        morl_env=None,
    ):

        self.env = env
        self.test_env = test_env
        if use_morl():
            self.morl_env: api.MorlEnv = morl_env
            self.morl_env_spec = self.morl_env.get_partial_spec()
            core.set_default_torch_device("cuda")
            # self.max_episode_steps = 108_000

        if use_morl():
            extu.manual_seed_morl(seed)
            self.env.seed(seed)
            self.test_env.seed(seed)
        else:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)
            self.test_env.seed(2**31 - 1 - seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu"
        )

        self.online_net = None
        self.target_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                memory_size,
                self.env.observation_space.shape,
                self.device,
                gamma,
                multi_step,
                beta_steps=beta_steps,
            )
        else:
            self.memory = LazyMultiStepMemory(
                memory_size,
                self.env.observation_space.shape,
                self.device,
                gamma,
                multi_step,
            )


        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, "model")
        self.summary_dir = os.path.join(log_dir, "summary")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma**multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        # if use_morl():
        #     return True
        return self.steps % self.update_interval == 0 and self.steps >= self.start_steps

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        if use_morl():
            return self.morl_env_spec.sample_actions()
        return self.env.action_space.sample()

    def exploit(self, state):
        # Act without randomness.
        if use_morl():
            state = (
                torch.as_tensor(state, dtype=torch.float32, device=self.device) / 255
            )
        else:
            state = torch.ByteTensor(state).unsqueeze(0).to(self.device).float() / 255.0
        with torch.no_grad():
            if use_morl():
                action = self.online_net(states=state).outputs.argmax()
            else:
                action = self.online_net.calculate_q(states=state).argmax()
        if use_morl():
            return action.view(-1, 1).cpu().numpy()
        return action.item()

    @abstractmethod
    def learn(self):
        pass

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(), os.path.join(save_dir, "online_net.pth")
        )
        torch.save(
            self.target_net.state_dict(), os.path.join(save_dir, "target_net.pth")
        )

    def load_models(self, save_dir):
        self.online_net.load_state_dict(
            torch.load(os.path.join(save_dir, "online_net.pth"))
        )
        self.target_net.load_state_dict(
            torch.load(os.path.join(save_dir, "target_net.pth"))
        )

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.0
        episode_steps = 0

        done = False
        if use_morl():
            state = self.morl_env.get_states_()
        else:
            state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better
            # performances.
            if use_morl():
                extu.reset_noise(self.online_net)
            else:
                self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore()
            elif use_morl():
                action = self.exploit(self.morl_env.get_states_())
            else:
                action = self.exploit(state)

            if use_morl():
                exp_ = self.morl_env.step_(action)
                next_state, reward, done = (
                    exp_.next_states.squeeze(),
                    exp_.rewards.item(),
                    exp_.dones.item(),
                )
            else:
                next_state, reward, done, _ = self.env.step(action)

            # To calculate efficiently, I just set priority=max_priority here.
            if use_morl():
                self.morl_memory.add_(exp_)
            else:
                self.memory.append(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)


        print(
            f"Episode: {self.episodes:<4}  "
            f"episode steps: {episode_steps:<4}  "
            f"return: {episode_return:<5.1f}"
        )

    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        if self.steps % self.eval_interval == 0:
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, "final"))
            self.online_net.train()

    def evaluate(self):
        self.online_net.eval()
        num_episodes = 0
        num_steps = 0
        total_return = 0.0
        pbar = tqdm.tqdm(total=self.num_eval_steps)
        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_random(eval=True):
                    action = self.explore()
                else:
                    if use_morl():
                        state = state[None, ...]
                    action = self.exploit(state)

                next_state, reward, done, _ = self.test_env.step(int(action))
                num_steps += 1
                pbar.update()
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return
            pbar.set_description(f"Eval: {total_return/num_episodes:<5.1f}")

            if num_steps > self.num_eval_steps:
                pbar.close()
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, "best"))

        # We log evaluation results along with training frames = 4 * steps.
        print("-" * 60)
        print(f"Num steps: {self.steps:<5}  " f"return: {mean_return:<5.1f}")
        print("-" * 60)

    def __del__(self):
        self.env.close()
        self.test_env.close()
