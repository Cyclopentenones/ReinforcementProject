import torch
import matplotlib.pyplot as plt
import heapq

class Packing:
    def __init__(self, width, height, items, device):
        self.width = width
        self.height = height
        self.items = items
        self.num_items = len(items)
        self.device = device
        self.frame = torch.zeros((self.height, self.width), device=self.device)  # Frame (H, W)

    def reset(self):
        self.frame = torch.zeros((self.height, self.width), device=self.device)
        self.placed_items = []
        self.remain_items = sorted([list(item) for item in self.items], key=lambda x: x[0] * x[1], reverse=True)  # Sort items by area

    def can_place(self, x, y, h, w):
        if x + h > self.height or y + w > self.width:
            return False
        if not torch.all(self.frame[x:x + h, y:y + w] == 0):
            return False
        return True

    def can_place_anywhere(self, h, w):
        for x in range(0, self.height - h + 1):
            for y in range(0, self.width - w + 1):
                if self.can_place(x, y, h, w):
                    return True
        return False

    def place(self, action, mode='Minimize Waste'):
        index, rotated = action

        # Validate item index
        if index >= len(self.items) or self.remain_items[index] == [0, 0]:
            return False, -1

        # Get item dimensions
        l, w = self.items[index]
        if rotated:
            l, w = w, l

        placed = False
        pos = None

        # Use a more efficient approach to find a placement
        for x in range(0, self.height - l + 1):
            for y in range(0, self.width - w + 1):
                if self.can_place(x, y, l, w):
                    self.frame[x:x + l, y:y + w] = 1  # Mark space as occupied
                    self.remain_items[index] = [0, 0]  # Mark item as placed
                    self.placed_items.append((index, x, y, l, w, rotated))
                    pos = (x, y)
                    placed = True
                    break
            if placed:
                break

        # Calculate reward if placed
        if placed:
            reward_S = l * w
            free_space = self.width * self.height - torch.sum(self.frame).item()
            reward_ratio = free_space / (self.width * self.height) if self.width * self.height > 0 else 0
            area_ratio = (l * w) / free_space if free_space > 0 else 0
            x, y = pos
            neighbors = torch.sum(self.frame[max(0, x - 1):x + l + 1, max(0, y - 1):y + w + 1]).item() - l * w
            connectedness = neighbors / (2 * (l + w)) if (2 * (l + w)) != 0 else 0

            if mode == "Minimize Waste":
                reward = (
                    reward_S * reward_ratio * 0.5 +
                    area_ratio * 0.3 +
                    min(l, w) / max(l, w) * 0.1 +
                    connectedness * 0.1
                )
            else:
                reward = reward_S * 0.5
        else:
            reward = -1  # Penalty for failure

        return placed, reward

    def is_done(self):
        return all(item == [0, 0] for item in self.remain_items)

    def get_state(self):
        remaining_items_matrix = torch.tensor(self.remain_items, device=self.device)  # (N, 2)
        state = (self.frame.unsqueeze(0), remaining_items_matrix)  # (H, W) -> (1, H, W)
        return state

    def get_valid_actions(self, action_space, max_items_considered=10):
        # Sort remaining items by area
        items_with_area = [(idx, self.items[idx][0] * self.items[idx][1]) for idx in range(len(self.remain_items)) if self.remain_items[idx] != [0, 0]]
        items_with_area.sort(key=lambda x: x[1], reverse=True)

        # Select top items to consider
        top_items = {i for i, _ in items_with_area[:max_items_considered]}

        valid_actions = []
        for idx, (index, rotated) in enumerate(action_space):
            if index in top_items:
                l, w = self.items[index]
                if rotated:
                    l, w = w, l
                if self.can_place_anywhere(l, w):
                    valid_actions.append(idx)
        return valid_actions

    def render(self):
        plt.figure(figsize=(10, 5))
        ax = plt.gca()

        for index, x, y, h, w, rotated in self.placed_items:
            color = 'blue' if rotated else 'pink'
            rect = plt.Rectangle((y, x), w, h, edgecolor='black', facecolor=color, alpha=0.5, linewidth=1)
            ax.add_patch(rect)
            plt.text(y + w / 2, x + h / 2, f"{h}×{w}", ha='center', va='center', fontsize=8)

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Packing')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.show()
