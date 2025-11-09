import jax
import jax.numpy as jnp
import optax
from flax import nnx
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_datasets = torchvision.datasets.CIFAR10(
    root='/home/rose/cifar10', 
    train=True, 
    download=True, 
    transform=transform
)
test_datasets = torchvision.datasets.CIFAR10(
    root='/home/rose/cifar10', 
    train=False, 
    download=True, 
    transform=transform
)

batch_size = 64
train_loader = DataLoader(
    train_datasets, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)
test_loader = DataLoader(
    test_datasets, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)

class SimpleCNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.linear = nnx.Linear(64 * 8 * 8, 10, rngs=rngs)
    
    def __call__(self, x, train=True):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = self.linear(x)
        return x

@nnx.jit
def train_step(model, optimizer, batch):
    images, labels = batch
    
    def loss_fn(model):
        logits = model(images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        
        pred_labels = jnp.argmax(logits, axis=-1)
        accuracy = (pred_labels == labels).mean()
        
        return loss, accuracy
    
    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return loss, accuracy

@nnx.jit
def eval_step(model, batch):
    images, labels = batch
    logits = model(images)
    pred_labels = jnp.argmax(logits, axis=-1)
    accuracy = (pred_labels == labels).mean()
    return accuracy

def print_accuracy(accuracy, prefix="准确度"):
    accuracy_percent = accuracy * 100
    print(f"{prefix}: {accuracy_percent:.2f}%")

def main():
    rng = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(rng)
    
    model = SimpleCNN(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))
    
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images_np = images.numpy().transpose(0, 2, 3, 1)
            labels_np = labels.numpy()
            
            images_jax = jnp.array(images_np)
            labels_jax = jnp.array(labels_np)
            
            loss, accuracy = train_step(model, optimizer, (images_jax, labels_jax))
            
            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: 损失 = {loss:.4f}, ", end="")
                print_accuracy(accuracy)
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        print(f"\n Epoch {epoch+1} 完成:")
        print(f"平均训练损失: {avg_loss:.4f}")
        print_accuracy(avg_accuracy, "平均训练准确度")
        
        test_accuracy = 0.0
        test_batches = 0
        
        for images, labels in test_loader:
            images_np = images.numpy().transpose(0, 2, 3, 1)
            labels_np = labels.numpy()
            
            images_jax = jnp.array(images_np)
            labels_jax = jnp.array(labels_np)
            
            accuracy = eval_step(model, (images_jax, labels_jax))
            test_accuracy += accuracy
            test_batches += 1
        
        avg_test_accuracy = test_accuracy / test_batches
        print_accuracy(avg_test_accuracy, "测试准确度")
        print("-" * 50)

if __name__ == "__main__":
    main()