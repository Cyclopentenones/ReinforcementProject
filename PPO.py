import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
class PPOAgent:
    def __init__(self, frame_shape, num_items, action_size, device, gamma=0.99, lr=1e-3, eps_clip=0.2, K_epochs=10):
        self.frame_shape = frame_shape
        self.num_items = num_items
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lr = lr
        self.K_epochs = K_epochs
        self.device = device  # CPU or GPU

        # PPO network for frame
        self.frame_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        # MLP for processing remaining items (num_items * 2)
        self.item_net = nn.Sequential(
            nn.Linear(num_items * 2, 128),
            nn.ReLU()
        ).to(device)

        # Combined size of frame_net and item_net
        self.combined_size = self._get_combined_size()

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        ).to(device)

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.frame_net.parameters()) + 
            list(self.item_net.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), lr=lr
        )
        self.policy_old = self.actor
        self.policy_old.load_state_dict(self.actor.state_dict())
        self.MseLoss = nn.MSELoss()

    def _get_combined_size(self):
        # Calculate the output size after passing through the networks
        with torch.no_grad():
            dummy_frame = torch.zeros(1, 1, *self.frame_shape).to(self.device)
            dummy_items = torch.zeros(1, self.num_items * 2).to(self.device)
            frame_features = self.frame_net(dummy_frame).shape[1]
            item_features = self.item_net(dummy_items).shape[1]
            return frame_features + item_features

    def select_action(self, state, valid_actions):
        frame, remaining_items = state
        frame_features = self.frame_net(frame.unsqueeze(0).unsqueeze(0).to(self.device))
        item_features = self.item_net(remaining_items.view(1, -1).float().to(self.device))
        combined_features = torch.cat((frame_features, item_features), dim=1)
        probs = self.actor(combined_features).squeeze(0)

        if not valid_actions:
            probs = torch.ones(self.action_size, device=self.device) / self.action_size
        else:
            mask = torch.zeros_like(probs)
            mask[valid_actions] = 1
            masked_probs = probs * mask
            masked_probs = masked_probs + 1e-8  # Add epsilon to avoid NaNs
            masked_probs = masked_probs / masked_probs.sum()

        dist = Categorical(masked_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(agent, memory, batch_size):
        total_loss = 0

        # Chuyển đổi danh sách thành tensor
        frame = torch.stack(memory.frame).to(agent.device).detach()
        remaining_items = torch.stack(memory.items).to(agent.device).detach()
        old_actions = torch.stack(memory.actions).to(agent.device).detach()
        old_logprobs = torch.stack(memory.log_probs).to(agent.device).detach()
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(agent.device)
        is_terminals = torch.tensor(memory.is_terminals, dtype=torch.float32).to(agent.device)

        num_samples = len(frame)
        for i in range(0, num_samples, batch_size):
            batch_frame = frame[i:i + batch_size]
            batch_remaining_items = remaining_items[i:i + batch_size]
            batch_actions = old_actions[i:i + batch_size]
            batch_rewards = rewards[i:i + batch_size]
            batch_is_terminals = is_terminals[i:i + batch_size]

            for _ in range(agent.K_epochs):  # Sử dụng K epochs
                # Đánh giá hành động và giá trị
                logprobs, state_values, dist_entropy = agent.evaluate(batch_frame, batch_remaining_items, batch_actions)

                # Tính TD target
                td_targets = batch_rewards[:-1].unsqueeze(1) + agent.gamma * state_values[1:].detach() * (1 - batch_is_terminals[1:].unsqueeze(1))
                td_targets = torch.cat((td_targets, torch.tensor([[0.0]])))

                # Tính lợi thế
                advantages = td_targets - state_values.detach()

                # PPO Clip Objective
                ratios = torch.exp(logprobs - old_logprobs[i:i + batch_size].detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * agent.MseLoss(state_values.squeeze(1), td_targets.squeeze(1)) - 0.01 * dist_entropy

                # Cập nhật mạng
                agent.optimizer.zero_grad()
                loss.mean().backward()
                agent.optimizer.step()
                total_loss += loss.mean().item()

        # Copy chính sách mới vào chính sách cũ
        agent.policy_old.load_state_dict(agent.actor.state_dict())
        return total_loss

    def evaluate(self, frame, remaining_items, actions):
        # Process frame and items qua các mạng tương ứng
        frame_features = self.frame_net(frame)  # (batch_size, frame_features)
        item_features = self.item_net(remaining_items.view(remaining_items.size(0), -1).float())  # (batch_size, item_features)
        combined_features = torch.cat((frame_features, item_features), dim=1)

        # Tính toán giá trị và xác suất hành động
        action_probs = self.actor(combined_features)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(combined_features)

        return action_logprobs, state_values, dist_entropy

