import torch
import torch.nn as nn
import torch.optim as optim
from PPO import PPOAgent
from Env import Packing
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_agent(agent, save_path):
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'gamma': agent.gamma,
        'eps_clip': agent.eps_clip,
        'lr': agent.lr,
        'K_epochs': agent.K_epochs
    }, save_path)

class Memory:
    def __init__(self): 
        self.frame = []
        self.items = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def update(self, frame, items, action, log_prob, reward, is_terminal): 
        self.frame.append(frame)
        self.items.append(items)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self): 
        self.frame = []
        self.items = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

def trainPPO(env, agent, memory, device, gamma=0.99, batch_size=32, update_mode='TD', save_path='ppo_model.pth', pbar=None):
    action_space = [(i, rotated) for i in range(env.num_items) for rotated in (True, False)]
    env.reset()
    state = env.get_state()
    total_reward = 0
    done = False
    sum_loss = 0
    while not done:
        valid_actions = env.get_valid_actions(action_space)
        if not valid_actions:
            break

        action_idx, log_prob = agent.select_action(state, valid_actions)
        action = action_space[action_idx]
        success, reward = env.place(action)
        if not success:
            reward = -1

        next_state = env.get_state()
        next_frame, next_remaining_items = next_state
        memory.update(next_frame, next_remaining_items, torch.tensor(action_idx), log_prob, reward, done)

        state = next_state
        total_reward += reward
        done = env.is_done()

        # Nếu chọn TD, cập nhật sau mỗi batch
        if update_mode == 'TD' and (len(memory.items) >= batch_size or done):
            sum_loss += agent.update(memory, batch_size)
            memory.clear()

        # Cập nhật thanh tiến trình nếu có
        if pbar:
            pbar.update(1)

    save_agent(agent, save_path)
    print(f"Model saved to {save_path}")
    return total_reward, sum_loss


if __name__ == '__main__':
    # Generate datasets for items
    num_sets = 250
    height, width = 100, 100
    sets = []
    for idx in range(num_sets):
        torch.manual_seed(23520932 + idx)
        mean_w, mean_h = 10.0, 10.0
        std_w, std_h = 3.0, 4.0
        num_items = 115 
        items = []
        for i in range(num_items): 
            w = max(1, int(torch.normal(mean=mean_w, std=std_w, size=()).item()))
            h = max(1, int(torch.normal(mean=mean_h, std=std_h, size=()).item()))
            items.append((w, h))
        sets.append(items)

    device = 'cpu'
    print('Training using: ', device)

    # Initialize agent and memory
    frame_shape = (height, width)
    agent = PPOAgent(
        frame_shape=frame_shape,
        num_items=len(sets[0]),
        action_size=2 * len(sets[0]),  # Account for rotation (True/False)
        device=device
    )
    episodes = 100
    rewards = []
    all_losses = []
    memory = Memory()
    batch_size = 32  # Set batch size
    with tqdm(total=episodes, desc="Training A2C") as episode_pbar:
        for episode in range(episodes):
            items = random.choice(sets)
            env = Packing(width, height, items, device)

            with tqdm(total=len(items), desc=f"Episode {episode+1}", leave=False) as batch_pbar:
                total_reward, sum_loss = trainPPO(env, agent, memory, device, gamma=0.99, pbar=batch_pbar)

            rewards.append(total_reward)
            all_losses.append(sum_loss)
            episode_pbar.update(1)
            print(f"Episode {episode+1}: Total Reward: {total_reward}")

    # Render final packing state
    env.render()

    # Plot reward over episodes
    plt.figure(figsize=(12, 6))

    # Vẽ reward qua các episodes
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')

    # Vẽ loss qua các batch (hoặc qua các episode nếu cần)
    plt.subplot(1, 2, 2)
    plt.plot(all_losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss over Batches')

    plt.tight_layout()
    plt.show()
