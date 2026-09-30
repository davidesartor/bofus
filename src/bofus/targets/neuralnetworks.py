from typing import Protocol, Callable
from jaxtyping import Float, Int, Array, Scalar, Key

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from optax.losses import softmax_cross_entropy_with_integer_labels as cross_entropy
import torchvision


class TestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class ResidualActivation(eqx.Module):
    """Activation as a module so its parameters stay traced instead of baked in as constants."""

    f: Callable

    def __call__(self, x: Scalar) -> Scalar:
        return self.f((x[None] + 1.0) / 2.0) + jax.nn.relu(x)


class MNIST(TestFunction):
    d: int = 1

    def __init__(
        self,
        seed: int = 0,
        n_runs: int = 5,
        batch_size: int = 100,
        width_size: int = 128,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 5,
    ):
        assert 10000 % batch_size == 0, "Batch size must divide 10000"
        # save parameters
        self.seed = seed
        self.batch_size = batch_size
        self.width_size = width_size
        self.depth = depth
        self.optimizer = optax.adam(lr)
        self.epochs = epochs
        self.n_runs = n_runs

        # load mnist dataset and preprocess (scale input to [0, 1])
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True
        )
        self.train_data: Float[Array, "n 28*28"] = jnp.array(train_dataset.data) / 255.0
        self.train_labels: Int[Array, "n"] = jnp.array(train_dataset.targets)
        self.test_data: Float[Array, "m 28*28"] = jnp.array(test_dataset.data) / 255.0
        self.test_labels: Int[Array, "m"] = jnp.array(test_dataset.targets)

        # compile once per instance, so repeated calls reuse the same executables
        self.jit_initialize = eqx.filter_jit(self.initialize)
        self.jit_fit = eqx.filter_jit(self.fit)
        self.jit_test = eqx.filter_jit(self.test)

    def __call__(self, f: Callable[[Float[Array, "1"]], Scalar]) -> Scalar:
        accuracy = []
        for key in jr.split(jr.key(self.seed), self.n_runs):
            key_init, key_fit = jr.split(key)
            network = self.jit_initialize(key_init, f)
            network, train_losses = self.jit_fit(network, key_fit)
            test_loss, test_accuracy = self.jit_test(network)
            accuracy.append(test_accuracy)
        return 1 - jnp.array(accuracy).mean()

    def initialize(self, key: Key, f: Callable[[Float[Array, "1"]], Scalar]):
        activation = ResidualActivation(f)
        return eqx.nn.MLP(
            in_size=28 * 28,
            out_size=10,
            width_size=self.width_size,
            depth=self.depth,
            activation=activation,
            scan=True,
            key=key,
        )

    def fit(self, network: eqx.nn.MLP, key: Key):
        @jax.value_and_grad
        def loss_fn(net_state, batch_data, batch_labels):
            network = eqx.combine(net_static, net_state)
            logits = jax.vmap(network)(batch_data)
            loss = cross_entropy(logits, batch_labels)
            return loss.mean()

        def train_step(state, inputs):
            batch_data, batch_labels = inputs
            net_state, opt_state = state
            loss, grads = loss_fn(net_state, batch_data, batch_labels)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            net_state = eqx.apply_updates(net_state, updates)
            return (net_state, opt_state), loss

        def train_epoch(state, key: Key):
            perm = jr.permutation(key, len(self.train_data))
            data = self.train_data[perm].reshape(-1, self.batch_size, 28 * 28)
            labels = self.train_labels[perm].reshape(-1, self.batch_size)
            state, train_loss = jax.lax.scan(train_step, state, (data, labels))
            return state, train_loss.mean()

        net_state, net_static = eqx.partition(network, eqx.is_inexact_array)
        opt_state = self.optimizer.init(net_state)
        (net_state, opt_state), train_losses = jax.lax.scan(
            train_epoch, (net_state, opt_state), jr.split(key, self.epochs)
        )
        network = eqx.combine(net_static, net_state)
        return network, train_losses

    def test(self, network: eqx.nn.MLP):
        def test_step(_, inputs):
            data, labels = inputs
            logits = jax.vmap(network)(data)
            loss = cross_entropy(logits, labels)
            acc = logits.argmax(axis=-1) == labels
            return _, (loss.mean(), acc.mean())

        data = self.test_data.reshape(-1, self.batch_size, 28 * 28)
        labels = self.test_labels.reshape(-1, self.batch_size)
        _, (test_loss, test_acc) = jax.lax.scan(test_step, None, (data, labels))
        return test_loss.mean(), test_acc.mean()
