import torch
import torch.nn as nn

class MultiplyByTwoModel(nn.Module):
    def __init__(self, hidden_size=10):
        super(MultiplyByTwoModel, self).__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    model_state = torch.load("saved_models/model_state_20250919_201925.pt", weights_only=False)
    model = MultiplyByTwoModel(hidden_size=10)
    model.load_state_dict(model_state)

    test_values = [1, 2, 5, -3, 10]
    print(f"\nТестування на конкретних значеннях після завантаження моделі:")
    for val in test_values:
        input_tensor = torch.FloatTensor([[val]])
        predicted = model(input_tensor).item()
        expected = val * 2
        print(f"Вхід: {val}, Передбачення: {predicted:.2f}, Очікується: {expected}")
