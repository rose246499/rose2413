import jax
import jax.numpy as jnp
import optax
from flax import nnx

class SimpleCNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
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

@nnx.jit  # 使用 nnx.jit
def simple_train_step(model, optimizer, batch):
    images, labels = batch
    
    def loss_fn(model):
        logits = model(images)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        
        # 计算准确度
        pred_labels = jnp.argmax(logits, axis=-1)
        true_labels = jnp.argmax(labels, axis=-1)
        accuracy = (pred_labels == true_labels).mean()
        
        return loss, accuracy
    
    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return loss, accuracy

def print_accuracy(accuracy, prefix="准确度"):
    """打印准确度百分比"""
    accuracy_percent = accuracy * 100
    print(f" {prefix}: {accuracy_percent:.2f}%")

def main():
    rng = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(rng)
    
    model = SimpleCNN(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))
  
    dummy_batch = (
        jnp.ones((2, 32, 32, 3)),
        jnp.eye(10)[:2] 
    )
    
    loss, accuracy = simple_train_step(model, optimizer, dummy_batch)
    
    print(f"训练损失: {loss:.4f}")
    print_accuracy(accuracy)
   
    for i in range(5):
        loss, accuracy = simple_train_step(model, optimizer, dummy_batch)
        print(f"第{i+1}次 - 损失: {loss:.4f}, ", end="")
        print_accuracy(accuracy, "准确度")

if __name__ == "__main__":
    main()
