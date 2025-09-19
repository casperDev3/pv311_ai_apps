import torch
import torch.nn as nn

# class MultiplyByTwoModel(nn.Module):
#     def __init__(self, hidden_size=10):
#         super(MultiplyByTwoModel, self).__init__()
#         self.hidden = nn.Linear(1, hidden_size)
#         self.relu = nn.ReLU()
#         self.output = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         x = self.hidden(x)
#         x = self.relu(x)
#         x = self.output(x)
#         return x

if __name__ == "__main__":
    model_full = torch.load("saved_models/model_full_20250919_205524.pt", weights_only=False)
    model_full.eval()

    test_values = [1, 2, 5, -3, 10]
    print(f"\nТестування на конкретних значеннях після завантаження моделі:")
    for val in test_values:
        input_tensor = torch.FloatTensor([[val]])
        predicted = model_full(input_tensor).item()
        expected = val * 2
        print(f"Вхід: {val}, Передбачення: {predicted:.2f}, Очікується: {expected}")
