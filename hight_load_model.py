import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import gc
import threading
import psutil
import time
import multiprocessing
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import classification_report
import random


# Конфігурація ресурсів системи
class ResourceConfig:
    def __init__(self,
                 max_cpu_usage=80,
                 max_memory_usage=85,
                 batch_size=32,
                 num_workers=2,
                 pin_memory=True,
                 mixed_precision=True,
                 gradient_accumulation=1,
                 checkpoint_freq=5):

        self.max_cpu_usage = max_cpu_usage
        self.max_memory_usage = max_memory_usage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mixed_precision = mixed_precision
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_freq = checkpoint_freq

        self._auto_adjust_parameters()

    def _auto_adjust_parameters(self):
        cpu_count = psutil.cpu_count(logical=False) if psutil.cpu_count(logical=False) else 2
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        if os.name == 'nt':
            self.num_workers = 0
        else:
            if self.num_workers == -1:
                self.num_workers = min(cpu_count, 4)

        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_memory_gb < 4:
                    self.batch_size = min(self.batch_size, 16)
                elif gpu_memory_gb < 8:
                    self.batch_size = min(self.batch_size, 32)
                else:
                    self.batch_size = min(self.batch_size, 64)  # Збільшуємо для кращої точності
            except Exception:
                self.batch_size = min(self.batch_size, 16)
        else:
            if memory_gb < 8:
                self.batch_size = min(self.batch_size, 16)

        if not torch.cuda.is_available():
            self.mixed_precision = False

        print(f"Автоматичні налаштування:")
        print(f"  Розмір батчу: {self.batch_size}")
        print(f"  Кількість воркерів: {self.num_workers}")
        print(f"  Змішана точність: {self.mixed_precision}")


# Клас для моніторингу ресурсів
class ResourceMonitor:
    def __init__(self, config):
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = defaultdict(list)

    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                self.resource_history['cpu'].append(cpu_percent)
                self.resource_history['memory'].append(memory_percent)

                if cpu_percent > self.config.max_cpu_usage or memory_percent > self.config.max_memory_usage:
                    print(f"Високе використання ресурсів: CPU {cpu_percent:.1f}%, RAM {memory_percent:.1f}%")
                    time.sleep(0.5)

                time.sleep(1.0)
            except Exception as e:
                print(f"Помилка моніторингу: {e}")
                break

    def get_resource_usage(self):
        try:
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.memory_allocated() > 0 else 0

            return {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'gpu_memory_gb': gpu_usage
            }
        except Exception:
            return {'cpu': 0, 'memory': 0, 'gpu_memory_gb': 0}


# Покращений датасет з аугментацією
class EnhancedCatDataset(Dataset):
    def __init__(self, cifar_dataset, max_samples=None, is_train=True, balance_ratio=1.0):
        self.data = []
        self.labels = []
        self.is_train = is_train

        cat_samples = []
        non_cat_samples = []

        print("Створення покращеного датасету котиків...")

        # Збираємо всі зразки
        for i, (img, label) in enumerate(cifar_dataset):
            if label == 3:  # Котик
                cat_samples.append((img, 1))
            else:  # Не котик
                non_cat_samples.append((img, 0))

        # Балансування датасету
        if max_samples:
            target_cats = min(len(cat_samples), max_samples // 2)
            target_non_cats = min(len(non_cat_samples), int(target_cats * balance_ratio))
        else:
            target_cats = len(cat_samples)
            target_non_cats = min(len(non_cat_samples), int(target_cats * balance_ratio))

        # Випадкова вибірка
        random.shuffle(cat_samples)
        random.shuffle(non_cat_samples)

        selected_cats = cat_samples[:target_cats]
        selected_non_cats = non_cat_samples[:target_non_cats]

        # Об'єднуємо та перемішуємо
        all_samples = selected_cats + selected_non_cats
        random.shuffle(all_samples)

        for img, label in all_samples:
            self.data.append(img)
            self.labels.append(label)

        print(f"Створено датасет: {len(selected_cats)} котиків, {len(selected_non_cats)} не-котиків")
        print(
            f"Співвідношення котики/не-котики: {len(selected_cats) / (len(selected_cats) + len(selected_non_cats)):.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Покращена архітектура з ResNet-подібними блоками
class AdvancedCatDetectionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, num_classes=2):
        super(AdvancedCatDetectionCNN, self).__init__()

        # Початковий шар
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-подібні блоки
        self.layer1 = self._make_layer(32, 64, 2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout_rate=dropout_rate)

        # Глобальний average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Класифікатор з dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Ініціалізація ваг
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []

        # Перший блок з можливою зміною розміру
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))

        # Решта блоків
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_rate))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ResNet-подібний блок
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# Покращена функція втрат з label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=2):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_label = self.smoothing / (self.num_classes - 1)

        pred = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(smooth_label)
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=1))


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Покращена функція тренування
def train_advanced_model(model, train_loader, test_loader, resource_monitor, scaler, scheduler,
                         criterion, optimizer, num_epochs=20, device='cpu', target_accuracy=95.0):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    resource_monitor.start_monitoring()

    best_test_acc = 0.0
    patience = 8  # Збільшуємо patience
    patience_counter = 0

    # Для early stopping з target accuracy
    target_reached = False

    try:
        for epoch in range(num_epochs):
            print(f"\nЕпоха [{epoch + 1}/{num_epochs}]")

            resources = resource_monitor.get_resource_usage()
            print(f"Ресурси - CPU: {resources['cpu']:.1f}%, RAM: {resources['memory']:.1f}%")
            if torch.cuda.is_available():
                print(f"GPU пам'ять: {resources['gpu_memory_gb']:.2f} GB")

            # Тренування
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    if scaler and resource_config.mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total_train += target.size(0)
                    correct_train += (predicted == target).sum().item()

                    if batch_idx % 50 == 0:
                        current_acc = 100 * correct_train / total_train if total_train > 0 else 0
                        print(
                            f'  Батч [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')

                except Exception as e:
                    print(f"Помилка в батчі {batch_idx}: {e}")
                    continue

            train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
            avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

            # Тестування
            model.eval()
            correct_test = 0
            total_test = 0
            test_loss = 0.0

            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for data, target in test_loader:
                    try:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                        if scaler and resource_config.mixed_precision and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                output = model(data)
                                loss = criterion(output, target)
                        else:
                            output = model(data)
                            loss = criterion(output, target)

                        test_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        total_test += target.size(0)
                        correct_test += (predicted == target).sum().item()

                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())

                    except Exception as e:
                        print(f"Помилка в тестуванні: {e}")
                        continue

            test_accuracy = 100 * correct_test / total_test if total_test > 0 else 0
            avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0

            # Збереження результатів
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f'  Тренувальна точність: {train_accuracy:.2f}%')
            print(f'  Тестова точність: {test_accuracy:.2f}%')
            print(f'  Тренувальна втрата: {avg_loss:.4f}')
            print(f'  Тестова втрата: {avg_test_loss:.4f}')

            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Learning rate: {current_lr:.6f}')

            # Перевірка досягнення цільової точності
            if test_accuracy >= target_accuracy:
                if not target_reached:
                    print(f"🎯 ДОСЯГНУТО ЦІЛЬОВУ ТОЧНІСТЬ {target_accuracy}%!")
                    target_reached = True

            # Early stopping та збереження
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                patience_counter = 0

                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_acc': best_test_acc,
                        'train_acc': train_accuracy,
                        'model_class': AdvancedCatDetectionCNN,
                    }, 'best_cat_detection_model.pth')
                    print(f"💾 Збережено найкращу модель з точністю {best_test_acc:.2f}%")
                except Exception as e:
                    print(f"Помилка збереження моделі: {e}")
            else:
                patience_counter += 1

            # Early stopping тільки після досягнення мінімальної кількості епох
            if patience_counter >= patience and epoch > 10:
                print(f"Early stopping на епосі {epoch + 1}")
                print(f"Найкраща точність: {best_test_acc:.2f}%")
                break

            # Якщо досягли цільової точності і пройшли достатньо епох
            if target_reached and epoch > 15 and patience_counter >= 3:
                print(f"Зупиняємо тренування - ціль досягнута!")
                break

            if scheduler:
                scheduler.step()

            cleanup_memory()
            print('-' * 60)

            # Детальна статистика кожні 5 епох
            if (epoch + 1) % 5 == 0:
                print("\n📊 ДЕТАЛЬНА СТАТИСТИКА:")
                if len(all_predictions) > 0 and len(all_targets) > 0:
                    from collections import Counter
                    pred_counter = Counter(all_predictions)
                    target_counter = Counter(all_targets)

                    print(f"  Передбачення - Котики: {pred_counter.get(1, 0)}, Не котики: {pred_counter.get(0, 0)}")
                    print(f"  Реальні - Котики: {target_counter.get(1, 0)}, Не котики: {target_counter.get(0, 0)}")

                if len(test_accuracies) >= 5:
                    recent_avg = sum(test_accuracies[-5:]) / 5
                    print(f"  Середня точність за останні 5 епох: {recent_avg:.2f}%")
                print("-" * 60)

    except KeyboardInterrupt:
        print("Тренування перервано користувачем")
    except Exception as e:
        print(f"Помилка під час тренування: {e}")
        import traceback
        traceback.print_exc()
    finally:
        resource_monitor.stop_monitoring()
        cleanup_memory()

    return train_losses, train_accuracies, test_accuracies, best_test_acc


def get_advanced_transforms(input_size=96):
    """Покращені трансформації для високої точності"""

    # Потужні аугментації для тренування
    transform_train = transforms.Compose([
        transforms.Resize((input_size + 20, input_size + 20)),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ])

    # Мінімальні трансформації для тестування
    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_test


def main():
    """Головна функція для досягнення 95% точності"""
    try:
        print("🎯 ТРЕНУВАННЯ МОДЕЛІ ДО 95% ТОЧНОСТІ")
        print("=" * 60)

        # Встановлення seed
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Покращена конфігурація
        global resource_config
        resource_config = ResourceConfig(
            max_cpu_usage=80,
            max_memory_usage=85,
            batch_size=64,  # Збільшуємо batch size
            num_workers=0 if os.name == 'nt' else 4,
            mixed_precision=torch.cuda.is_available(),
            gradient_accumulation=1
        )

        resource_monitor = ResourceMonitor(resource_config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'Використовується пристрій: {device}')
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'GPU пам\'ять: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')

        # Покращені трансформації
        transform_train, transform_test = get_advanced_transforms(input_size=96)

        # Завантаження CIFAR-10
        print("\nЗавантаження CIFAR-10 датасету...")
        try:
            cifar_train = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
            cifar_test = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        except Exception as e:
            print(f"Помилка завантаження CIFAR-10: {e}")
            return

        # Створення збалансованих датасетів
        train_dataset = EnhancedCatDataset(cifar_train, max_samples=8000, is_train=True, balance_ratio=1.2)
        test_dataset = EnhancedCatDataset(cifar_test, max_samples=2000, is_train=False, balance_ratio=1.0)

        print(f"Тренувальний датасет: {len(train_dataset)} зразків")
        print(f"Тестовий датасет: {len(test_dataset)} зразків")

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=resource_config.batch_size,
            shuffle=True,
            num_workers=resource_config.num_workers,
            pin_memory=resource_config.pin_memory and torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=resource_config.num_workers > 0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=resource_config.batch_size,
            shuffle=False,
            num_workers=resource_config.num_workers,
            pin_memory=resource_config.pin_memory and torch.cuda.is_available(),
            persistent_workers=resource_config.num_workers > 0
        )

        # Покращена модель
        model = AdvancedCatDetectionCNN(dropout_rate=0.3, num_classes=2).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n🔧 МОДЕЛЬ:")
        print(f"  Загальна кількість параметрів: {total_params:,}")
        print(f"  Тренувальні параметри: {trainable_params:,}")

        # Покращена функція втрат і оптимізатор
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=2)

        # Використовуємо AdamW з weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Покращений scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        scaler = torch.cuda.amp.GradScaler() if resource_config.mixed_precision and torch.cuda.is_available() else None

        print(f"\n🚀 ПОЧАТОК ТРЕНУВАННЯ:")
        print(f"  Цільова точність: 95%")
        print(f"  Максимум епох: 30")
        print(f"  Розмір батчу: {resource_config.batch_size}")
        print(f"  Змішана точність: {resource_config.mixed_precision}")
        print("-" * 60)

        # Тренування до 95%
        train_losses, train_accs, test_accs, best_acc = train_advanced_model(
            model, train_loader, test_loader, resource_monitor, scaler, scheduler,
            criterion, optimizer, num_epochs=30, device=device, target_accuracy=95.0
        )

        print(f"\n🏁 ТРЕНУВАННЯ ЗАВЕРШЕНО!")
        if test_accs:
            print(f"  Фінальна тестова точність: {test_accs[-1]:.2f}%")
            print(f"  Найкраща тестова точність: {best_acc:.2f}%")

            if best_acc >= 95.0:
                print("🎉 ДОСЯГНУТО 95% ТОЧНОСТІ!")
            else:
                print(f"⚠️  Не вдалося досягти 95%, найкраща: {best_acc:.2f}%")

        # Покращена візуалізація
        if train_losses and train_accs and test_accs:
            plt.figure(figsize=(15, 10))

            # Графік втрат
            plt.subplot(2, 3, 1)
            plt.plot(train_losses, 'b-', label='Тренувальна втрата', linewidth=2)
            plt.title('Втрата під час тренування', fontsize=14)
            plt.xlabel('Епоха')
            plt.ylabel('Втрата')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Графік точності
            plt.subplot(2, 3, 2)
            epochs = range(1, len(train_accs) + 1)
            plt.plot(epochs, train_accs, 'b-o', label='Тренувальна', linewidth=2, markersize=4)
            plt.plot(epochs, test_accs, 'r-s', label='Тестова', linewidth=2, markersize=4)
            plt.axhline(y=95, color='g', linestyle='--', label='Ціль 95%', linewidth=2)
            plt.title('Точність моделі', fontsize=14)
            plt.xlabel('Епоха')
            plt.ylabel('Точність (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(40, 100)

            # Інформація про модель
            plt.subplot(2, 3, 3)
            model_info = [
                f"Параметри: {total_params:,}",
                f"Розмір батчу: {resource_config.batch_size}",
                f"Найкраща точність: {best_acc:.2f}%",
                f"Пристрій: {device}",
                f"Змішана точність: {resource_config.mixed_precision}",
                f"Епох натреновано: {len(train_accs)}"
            ]
            plt.text(0.1, 0.5, '\n'.join(model_info), fontsize=12,
                     verticalalignment='center', transform=plt.gca().transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Інформація про модель', fontsize=14)
            plt.axis('off')

            # Прогрес до цілі
            plt.subplot(2, 3, 4)
            if test_accs:
                max_acc = max(test_accs)
                progress = min(max_acc / 95.0 * 100, 100)
                colors = ['red' if acc < 95 else 'green' for acc in test_accs]
                plt.bar(epochs, test_accs, color=colors, alpha=0.7)
                plt.axhline(y=95, color='orange', linestyle='-', linewidth=3, label='Ціль 95%')
                plt.title(f'Прогрес до цілі: {progress:.1f}%', fontsize=14)
                plt.xlabel('Епоха')
                plt.ylabel('Точність (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)

            # Статистика тренування
            plt.subplot(2, 3, 5)
            if len(test_accs) > 5:
                improvement = test_accs[-1] - test_accs[0]
                best_epoch = test_accs.index(max(test_accs)) + 1
                avg_last_5 = sum(test_accs[-5:]) / 5

                stats_text = [
                    f"Покращення: {improvement:+.2f}%",
                    f"Найкраща епоха: {best_epoch}",
                    f"Середнє (останні 5): {avg_last_5:.2f}%",
                    f"Стандартне відхилення: {np.std(test_accs):.2f}",
                    "",
                    "Досягнення:",
                    f"• >90%: {'Так' if max(test_accs) > 90 else 'Ні'}",
                    f"• >95%: {'Так' if max(test_accs) > 95 else 'Ні'}",
                ]

                plt.text(0.1, 0.5, '\n'.join(stats_text), fontsize=11,
                         verticalalignment='center', transform=plt.gca().transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            plt.title('Статистика тренування', fontsize=14)
            plt.axis('off')

            # Рекомендації для покращення
            plt.subplot(2, 3, 6)
            if best_acc < 95:
                recommendations = [
                    "Рекомендації для покращення:",
                    "",
                    "• Збільшити кількість епох",
                    "• Зменшити learning rate",
                    "• Додати більше аугментацій",
                    "• Збільшити розмір датасету",
                    "• Використати transfer learning",
                    "• Налаштувати weight decay",
                    "• Експериментувати з dropout",
                    "• Спробувати різні архітектури"
                ]
                color = "lightcoral"
            else:
                recommendations = [
                    "Відмінний результат!",
                    "",
                    "Досягнуто цільову точність 95%",
                    "",
                    "Модель готова до використання:",
                    "• Збережено у best_model.pth",
                    "• Можна використовувати для",
                    "  розпізнавання котиків",
                    "• Рекомендується додатково",
                    "  протестувати на реальних даних"
                ]
                color = "lightgreen"

            plt.text(0.1, 0.5, '\n'.join(recommendations), fontsize=10,
                     verticalalignment='center', transform=plt.gca().transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            plt.title('Рекомендації', fontsize=14)
            plt.axis('off')

            plt.suptitle(f'Результати тренування до 95% точності (досягнуто: {best_acc:.2f}%)',
                         fontsize=16, y=0.98)
            plt.tight_layout()
            plt.show()

        # Тестування на реальних зображеннях
        print(f"\n🧪 ТЕСТУВАННЯ НА РЕАЛЬНИХ ЗОБРАЖЕННЯХ:")
        test_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg"
        ]

        model.eval()
        for i, url in enumerate(test_urls[:2]):  # Тестуємо лише 2 зображення
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')

                input_tensor = transform_test(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    if scaler and resource_config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model(input_tensor)
                    else:
                        output = model(input_tensor)

                    probabilities = torch.softmax(output, dim=1)
                    confidence_scores = probabilities.cpu().numpy()[0]
                    predicted = torch.argmax(probabilities, dim=1).item()

                is_cat = predicted == 1
                confidence = confidence_scores[predicted] * 100

                result_text = "КОТИК" if is_cat else "НЕ КОТИК"
                print(f"  Зображення {i + 1}: {result_text} (впевненість: {confidence:.1f}%)")

            except Exception as e:
                print(f"  Помилка тестування зображення {i + 1}: {e}")

        cleanup_memory()

        print(f"\n✅ ПРОГРАМА ЗАВЕРШЕНА УСПІШНО!")
        print(f"   Найкраща досягнута точність: {best_acc:.2f}%")

        if best_acc >= 95:
            print(f"   🎉 Ціль досягнута! Модель збережена.")
        else:
            print(f"   ⚠️ Для досягнення 95% рекомендується:")
            print(f"      - Збільшити кількість епох до 40-50")
            print(f"      - Використати більший датасет")
            print(f"      - Налаштувати гіперпараметри")

    except KeyboardInterrupt:
        print("\nПрограма перервана користувачем")
    except Exception as e:
        print(f"Критична помилка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_memory()


if __name__ == '__main__':
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    multiprocessing.freeze_support()
    main()