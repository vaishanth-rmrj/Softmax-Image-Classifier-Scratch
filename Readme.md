## What is image classification?

Imagine the classical example: You are given a set of images each of which either depicts a cat or a dog. Instead of labeling the pictures all on your own, you want to use an algorithm to do the work for you: It "looks" at the whole picture and outputs probabilities for each of the classes it was trained on.

This is usually made possible through training neural networks, which we describe in more detail in other articles. (Note: There are other techniques but they do not play a role in practice due to performance.) As in other applications of supervised learning, the network is fed with a sufficient training data – namely labeled images of cats and dogs.

What happens in between the image and output is somewhat obscure and we are going into greater detail in other posts. But in simple terms, most networks break down the image into abstract shapes and colors, which are used to form a hypothesis regarding the image's content.


### Image Classification Techniques

1. K Nearest Neighbor
2. Softmax
3. Support Vector Machines
4. Artificial Neural Networks
5. Convolutional Neural Networks

### Image Classification with Softmax

While hinge loss is quite popular, you’re more likely to run into cross-entropy loss and Softmax classifiers in the context of Deep Learning and Convolutional Neural Networks.

Why is Softmax?

Softmax classifiers give you probabilities for each class label while hinge loss gives you the margin.

It’s much easier for us as humans to interpret probabilities rather than margin scores (such as in hinge loss and squared hinge loss).


### The working in a little more detail is as follows

The Softmax classifier is a generalization of the binary form of Logistic Regression. Just like in hinge loss or squared hinge loss, our mapping function f is defined such that it takes an input set of data x and maps them to the output class labels via a simple (linear) dot product of the data x and weight matrix W:

f(x_{i}, W) = Wx_{i}

However, unlike hinge loss, we interpret these scores as unnormalized log probabilities for each class label — this amounts to swapping out our hinge loss function with cross-entropy loss:

L_{i} = -log(e^{s_{y_{i}}} / \sum_{j} e^{s_{j}})

So, how did I arrive here? Let’s break the function apart and take a look.

To start, our loss function should minimize the negative log likelihood of the correct class:

L_{i} = -log P(Y=y_{i}|X=x_{i})

This probability statement can be interpreted as:

P(Y=k|X=x_{i}) = e^{s_{y_{i}}} / \sum_{j} e^{s_{j}}

Where we use our standard scoring function form:

s = f(x_{i}, W)

As a whole, this yields our final loss function for a single data point, just like above:

L_{i} = -log(e^{s_{y_{i}}} / \sum_{j} e^{s_{j}})

Note: Your logarithm here is actually base e (natural logarithm) since we are taking the inverse of the exponentiation over e earlier.

The actual exponentiation and normalization via the sum of exponents is our actual Softmax function. The negative log yields our actual cross-entropy loss.

Just as in hinge loss or squared hinge loss, computing the cross-entropy loss over an entire dataset is done by taking the average:

L = \frac{1}{N} \sum^{N}_{i=1} L_{i}