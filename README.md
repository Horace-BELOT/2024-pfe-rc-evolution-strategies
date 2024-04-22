
# Echo State Network & Evolution Strategies

The goal of this project is to experiment with the training of an [ESN](http://www.scholarpedia.org/article/Echo_state_network)) using Evolution Strategies. Usually, only the output layer of an ESN is trained, using a simple SGD or PINV. It is also possible to tune the hyper-parameters of the reservoir to increase performances a bit further.

What we attempt to do here is to train the input layer using Evolution Strategies.

## Echo State Network

Here is a simple code snippet to train and test an ESN on MNIST :
```Python
# Load data
(x_train, y_train), (x_test, y_test)  =  load_mnist()

# Initialize ESN
esn  =  ESN(
	n_inputs=28*28,
	n_outputs=10,
	spectral_radius=0.8,
	n_reservoir=500,
	sparsity=0.5,
	silent=False,
	input_scaling=0.7,
	feedback_scaling=0,
	wash_out=25,
	learn_method="pinv",
)
# Train and predict on train set
pred_train  =  esn.fit(x_train, y_train)
# Predict on test dataset
pred_test  =  esn.predict(x_test, continuation=False)

# Compute accuracies
train_acc  =  accuracy(pred_train, y_train)
test_acc  =  accuracy(pred_test, y_test)

# Print results
print(f"Training accuracy: {100*train_acc:.2f}%")
print(f"Testing accuracy: {100*test_acc:.2f}%")
```
Results when executed :
```
Training accuracy: 86.32%
Testing accuracy: 86.35%
```

## Natural Evolution Strategy

This is an example of how to find a mystery matrix by minimizing the L2 distance between our matrix and a mystery matrix using NES.

```Python
n: int = 20
p: int = 80
# Creating mystery array we will try to converge to
target_array  = (upper_bound - lower_bound) * np.random.rand(n, p) + lower_bound
# Creating starting array, same size but full of zeros
base_array  =  np.zeros_like(target_array, dtype=float)

# Defining reward function that we will try to maximize: -distance(w, mystery)
reward_func = lambda x: - np.linalg.norm(target_array - x)/(n * p)

nes  =  NES(
	w=base_array,
	f=reward_func,
	pop=25,
	sigma=5 * 10 ** (-1),
	alpha=5 * 10 ** (-1)
)
loss_array  =  nes.optimize(n_iter=500, silent=False, graph=True)
plt.plot(-np.log10(-loss_array), label="loss")
plt.show()
```
