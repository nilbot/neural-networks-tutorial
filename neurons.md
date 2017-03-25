# Idealism
we want the neurons to fire with spikes, sharp and decisive, it's hard to find known and good mathematical model to represent this. (in continuous space)

## Linear Neurons
$$ y = b + \sum_i{x_iw_i} $$

* no sense of activation

## Binary Threshold Neurons
step function wrapped around a linear neurons

* not analytic

## Sigmoid Neurons

$$ y = \frac{1}{1+e^{-z}} $$
where $z$ is $b+\sum_i{x_iw_i}$.

# Learning
neurons' behaviours depend on he value of their parameter and stuff.

## Hebbian
says the repeatedly co-consipirators will strengthen their link overtime

## Rosenblatt
$$ z_j = \sum_i{w_{ji}x_i} $$
and 
$$ y = \begin{cases}
0, & \text{if} \\
1, & \text{else} 
\end{cases}
$$


## PAC Learning
PAC for probabal accurate

## Multi-layer perceptrons
